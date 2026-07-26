"""FastAPI app serving the interactive book map.

The store is opened read-only so serving can never contend with an ingest.
Layout is computed per request over the induced subgraph around the seeds and
results; the full graph is never laid out.

Two things are done once at startup rather than per request: the store is opened,
and the graph is projected into SciPy. Projecting tens of millions of edges takes
long enough that doing it per query would make the UI unusable, and neither the
store nor the projection is mutated by serving.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from bookmap.bridges import BridgesUnavailable, find_bridges
from bookmap.config import LayoutConfig, RecommendConfig
from bookmap.connections import ConnectionsUnavailable, find_connections
from bookmap.graph.connect import DEFAULT_MAX_HOPS, OBJECTIVES, PRODUCT
from bookmap.graph.layout import force_atlas2, normalize_positions
from bookmap.recommend import recommend as run_recommend
from bookmap.store.db import Store
from bookmap.store.projection import GraphProjection, project

STATIC = Path(__file__).with_name("static")

VIEWPORT = (1000.0, 700.0)
"""Layout viewport. The client rescales to its own canvas, but positions are
computed in a fixed space so a given subgraph always has the same shape."""


class RecommendRequest(BaseModel):
    seeds: list[str] = Field(..., min_length=1)
    n: int = Field(20, ge=1, le=200)
    beta: float = Field(0.25, ge=0.0, le=1.0)
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    exclude_same_author: bool = False


class SubgraphRequest(BaseModel):
    seeds: list[str] = Field(..., min_length=1)
    n: int = Field(40, ge=1, le=400)
    include_neighbors: bool = True
    include_connections: bool = True
    """Add the Steiner skeleton joining the seeds to the laid-out subgraph.

    On by default because without it the connecting books are *absent*, not
    merely unhighlighted: selection by relevance excludes exactly the nodes that
    would join two genres, so the lobes cannot be drawn together at all.
    """


def _book_json(book) -> dict[str, Any]:  # noqa: ANN001 - bookmap.models.Book
    return {
        "work_id": book.work_id,
        "title": book.title,
        "authors": list(book.authors),
        "year": book.year,
        "series": book.series,
        "avg_rating": book.avg_rating,
        "ratings_count": book.ratings_count,
        "subjects": list(book.subjects),
    }


def _recommendation_json(rec) -> dict[str, Any]:  # noqa: ANN001 - bookmap.models.Recommendation
    return {
        "work_id": rec.work_id,
        "title": rec.title,
        "authors": list(rec.authors),
        "score": rec.score,
        "raw_ppr": rec.raw_ppr,
        "degree": rec.degree,
        "community": rec.community,
        "explanations": [
            {
                "seed_id": exp.seed_id,
                "seed_title": exp.seed_title,
                "path": list(exp.path),
                "hops": exp.hops,
                "strength": exp.strength,
            }
            for exp in rec.explanations
        ],
    }


def build_subgraph_payload(
    seed_ids: list[str],
    scored: dict[str, float],
    projection: GraphProjection,
    store: Store,
    *,
    include_neighbors: bool = True,
    required: list[str] | None = None,
    config: LayoutConfig | None = None,
) -> dict[str, Any]:
    """Assemble the node/edge/position JSON the canvas renderer consumes.

    ``scored`` maps recommendation ids to their score; seeds carry no score.
    Neighbours of seeds are pulled in as context so the map shows a
    neighbourhood rather than a disconnected scatter of results.

    ``required`` nodes are included whatever their score -- the connectors on the
    Steiner skeleton, which by construction rank nowhere near the top and would
    otherwise be trimmed away. They are ordered directly after the seeds because
    the trim below is a prefix cut.

    Trimmed to ``config.max_nodes`` before layout: the force simulation is
    O(n^2) per iteration and, more importantly, a few hundred labelled circles is
    already past what anyone can read.
    """
    # Stronger repulsion and a longer run than the defaults: these subgraphs are
    # dense (books in a genre are near-cliques), and at the default settings the
    # clusters never get far enough apart to be legible as separate regions.
    config = config or LayoutConfig(iterations=500, scaling_ratio=14.0)

    members: list[str] = []
    seen: set[str] = set()
    # Seeds first, then results, then context: the trim below is a prefix cut, so
    # ordering here is what decides who survives it.
    ordered = [
        *seed_ids,
        *(required or []),
        *sorted(scored, key=lambda key: -scored[key]),
    ]
    for work_id in ordered:
        if work_id in projection.index and work_id not in seen:
            seen.add(work_id)
            members.append(work_id)

    if include_neighbors:
        # Seeds only, not every result. Expanding all of them pulls in each
        # recommendation's hub neighbours too, and the graph collapses into a
        # hairball where position conveys nothing -- which matters more than
        # usual here, because position is what carries cluster identity on this
        # map (colour cannot: nine communities have no accessible hue set).
        for work_id in seed_ids:
            if work_id not in projection.index:
                continue
            for neighbor, _weight in store.neighbors(work_id, limit=6):
                if neighbor in projection.index and neighbor not in seen:
                    seen.add(neighbor)
                    members.append(neighbor)

    members = members[: config.max_nodes]
    if not members:
        return {"nodes": [], "edges": [], "community_labels": {}}

    sub = projection.subgraph(members)
    positions = normalize_positions(
        force_atlas2(sub, config), width=VIEWPORT[0], height=VIEWPORT[1], padding=48.0
    )

    seed_set = set(seed_ids)
    books = store.get_books(sub.work_ids)
    degree = sub.degree()

    nodes = []
    for row, work_id in enumerate(sub.work_ids):
        book = books.get(work_id)
        nodes.append(
            {
                "work_id": work_id,
                "title": book.title if book is not None else work_id,
                "authors": list(book.authors) if book is not None else [],
                "x": float(positions[row][0]),
                "y": float(positions[row][1]),
                "community": store.node_community(work_id),
                "is_seed": work_id in seed_set,
                "score": scored.get(work_id),
                "degree": int(degree[row]),
                # Role drives colour, and it is the one thing position cannot
                # show: which of these books the user named, which the graph
                # proposed, and which are only context.
                "role": (
                    "seed"
                    if work_id in seed_set
                    else "recommendation"
                    if work_id in scored
                    else "context"
                ),
            }
        )

    adjacency = sub.adjacency.tocoo()
    edges = [
        {
            "src": sub.work_ids[int(i)],
            "dst": sub.work_ids[int(j)],
            "weight": float(weight),
        }
        # Upper triangle only: the adjacency is symmetric, so both orientations
        # would draw every line twice.
        for i, j, weight in zip(adjacency.row, adjacency.col, adjacency.data, strict=True)
        if int(i) < int(j) and weight != 0
    ]

    present = {node["community"] for node in nodes if node["community"] is not None}
    labels = store.community_labels()
    community_labels = {
        str(community): labels.get(community, f"cluster {community}") for community in present
    }

    return {"nodes": nodes, "edges": edges, "community_labels": community_labels}


def create_app(db_path: str = "bookmap.duckdb") -> FastAPI:
    """Build the app.

    Endpoints:
      ``GET  /``                       the map UI
      ``GET  /api/search?q=``          typeahead seed resolution
      ``POST /api/recommend``          ranked recommendations with explanations
      ``POST /api/subgraph``           nodes with layout positions, edges, communities
      ``GET  /api/book/{work_id}``     one book's details
      ``GET  /api/book/{id}/neighbors``click-to-expand
      ``GET  /api/bridges``            books linking two reading communities
      ``GET  /api/connections``        the route joining a set of books
      ``GET  /api/stats``              graph summary

    Unknown seeds return 200 with them listed under ``unresolved`` rather than
    an error: a partially-resolvable seed set is still useful.
    """
    store = Store.open(db_path, read_only=True)
    projection = project(store)
    # DuckDB connections are not safe to share across threads, and FastAPI runs
    # sync endpoints in a threadpool. One lock is ample: queries are milliseconds
    # and this is a local single-user tool.
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        store.close()

    app = FastAPI(title="bookmap", docs_url="/api/docs", lifespan=lifespan)
    app.state.store = store
    app.state.projection = projection

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/static/{name}")
    def static_file(name: str) -> HTMLResponse:
        path = (STATIC / name).resolve()
        # Resolve and confine: a name like "../../secrets" must not escape.
        if not path.is_file() or STATIC.resolve() not in path.parents:
            raise HTTPException(status_code=404, detail="not found")
        media = "text/css" if name.endswith(".css") else "application/javascript"
        return HTMLResponse(path.read_text(encoding="utf-8"), media_type=media)

    @app.get("/api/search")
    def search(q: str = Query("", description="Title fragment."), limit: int = 10) -> dict[str, Any]:
        if not q.strip():
            # An empty box is not an error; it simply has nothing to offer yet.
            return {"results": []}
        with lock:
            found = store.search_titles(q, limit=limit)
        return {
            "results": [
                {"work_id": b.work_id, "title": b.title, "authors": list(b.authors)}
                for b in found
            ]
        }

    @app.post("/api/recommend")
    def recommend_endpoint(request: RecommendRequest) -> dict[str, Any]:
        config = RecommendConfig(
            hub_beta=request.beta,
            mmr_lambda=request.mmr_lambda,
            exclude_same_author=request.exclude_same_author,
        )
        with lock:
            result = run_recommend(
                request.seeds, store, projection, n=request.n, config=config
            )
        return {
            "recommendations": [_recommendation_json(r) for r in result.recommendations],
            "seeds": [
                {"query": m.query, "work_id": m.work_id, "title": m.title,
                 "authors": list(m.authors), "confidence": m.confidence}
                for m in result.seeds
            ],
            "unresolved": list(result.unresolved),
            "ambiguous": [
                {"query": m.query, "alternatives": [list(alt) for alt in m.alternatives]}
                for m in result.ambiguous
            ],
        }

    @app.post("/api/subgraph")
    def subgraph_endpoint(request: SubgraphRequest) -> dict[str, Any]:
        with lock:
            result = run_recommend(request.seeds, store, projection, n=request.n)
            seed_ids = [m.work_id for m in result.seeds if m.work_id]
            scored = {r.work_id: r.score for r in result.recommendations}

            route: dict[str, Any] = {}
            if request.include_connections:
                route = find_connections(store, projection, seeds=request.seeds)

            skeleton_nodes = [
                node["work_id"]
                for skeleton in route.get("skeletons", [])
                for node in skeleton["nodes"]
            ]
            payload = build_subgraph_payload(
                seed_ids,
                scored,
                projection,
                store,
                include_neighbors=request.include_neighbors,
                required=skeleton_nodes,
            )

        drawn = {node["work_id"] for node in payload["nodes"]}
        payload["unresolved"] = list(result.unresolved)
        payload["connectors"] = [
            node["work_id"]
            for skeleton in route.get("skeletons", [])
            for node in skeleton["connectors"]
            if node["work_id"] in drawn
        ]
        # Confined to what was actually laid out. An edge naming a node the client
        # never received is silently dropped by the renderer, which reads as a
        # broken route rather than a trimmed one.
        payload["skeleton_edges"] = [
            {"src": edge["source"], "dst": edge["target"], "weight": edge["weight"]}
            for skeleton in route.get("skeletons", [])
            for edge in skeleton["edges"]
            if edge["source"] in drawn and edge["target"] in drawn
        ]
        payload["routes"] = [
            {"titles": leg["titles"], "hops": leg["hops"], "strength": leg["strength"]}
            for skeleton in route.get("skeletons", [])
            for leg in skeleton["legs"]
        ]
        return payload

    @app.get("/api/book/{work_id}")
    def book_detail(work_id: str) -> dict[str, Any]:
        with lock:
            book = store.get_book(work_id)
            community = store.node_community(work_id)
        if book is None:
            raise HTTPException(status_code=404, detail=f"no such book: {work_id}")
        payload = _book_json(book)
        payload["community"] = community
        return payload

    @app.get("/api/book/{work_id}/neighbors")
    def book_neighbors(work_id: str, limit: int = 25) -> dict[str, Any]:
        with lock:
            if store.get_book(work_id) is None:
                raise HTTPException(status_code=404, detail=f"no such book: {work_id}")
            pairs = store.neighbors(work_id, limit=limit)
            books = store.get_books([neighbor for neighbor, _ in pairs])
        return {
            "work_id": work_id,
            "neighbors": [
                {
                    "work_id": neighbor,
                    "title": books[neighbor].title if neighbor in books else neighbor,
                    "authors": list(books[neighbor].authors) if neighbor in books else [],
                    "weight": weight,
                }
                for neighbor, weight in pairs
            ],
        }

    @app.get("/api/bridges")
    def bridges_endpoint(
        seeds: str = Query("", description="Comma-separated titles to restrict to."),
        top_n: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        """Books linking two otherwise-separate reading communities.

        A missing community partition is a 409 rather than a 500: the database is
        readable, the graph simply has not been built, and that is the caller's to
        fix by running ``bookmap build``.
        """
        wanted = [part.strip() for part in seeds.split(",") if part.strip()]
        with lock:
            try:
                return find_bridges(store, projection, seeds=wanted or None, top_n=top_n)
            except BridgesUnavailable as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/connections")
    def connections_endpoint(
        seeds: str = Query("", description="Comma-separated titles to join."),
        max_hops: int = Query(DEFAULT_MAX_HOPS, ge=1, le=20),
        objective: str = Query(PRODUCT, description='"product" or "widest".'),
        min_edge_weight: float = Query(0.0, ge=0.0, le=1.0),
    ) -> dict[str, Any]:
        """The minimal structure joining the seeds, for the map's overlay.

        Fewer than two resolvable seeds is a 200 with no skeletons rather than an
        error: one book is a legitimate query that simply has no route in it, and
        the client falls back to the neighbourhood view. An empty ``seeds`` is a
        422, because that is a malformed request rather than an empty answer.
        """
        wanted = [part.strip() for part in seeds.split(",") if part.strip()]
        if not wanted:
            raise HTTPException(status_code=422, detail="seeds must name at least one book")
        if objective not in OBJECTIVES:
            raise HTTPException(
                status_code=422,
                detail=f"unknown objective {objective!r}; expected one of {OBJECTIVES}",
            )
        with lock:
            try:
                return find_connections(
                    store,
                    projection,
                    seeds=wanted,
                    max_hops=max_hops,
                    objective=objective,
                    min_edge_weight=min_edge_weight,
                )
            except ConnectionsUnavailable as exc:
                # Readable database, unbuilt graph: the caller's to fix with
                # `bookmap build`, so 409 rather than 500.
                raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        with lock:
            counts = store.counts()
            labels = store.community_labels()
        degree = projection.degree()
        return {
            **counts,
            "nodes": projection.n_nodes,
            "edges": projection.n_edges,
            "communities_named": len(labels),
            "mean_degree": float(degree.mean()) if degree.size else 0.0,
            "max_degree": int(degree.max()) if degree.size else 0,
        }

    return app


def serve(db_path: str = "bookmap.duckdb", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the app under uvicorn. Used by ``bookmap web``."""
    import uvicorn

    uvicorn.run(create_app(db_path), host=host, port=port, log_level="info")


__all__ = ["build_subgraph_payload", "create_app", "serve"]
