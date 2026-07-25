"""``bookmap`` command line interface.

    bookmap ingest demo
    bookmap ingest goodreads-ucsd ~/Downloads/goodreads_books.json.gz
    bookmap build --min-weight 0.05
    bookmap recommend --seeds "Dune,Foundation,Hyperion" -n 20
    bookmap map --seeds "Dune,Foundation" --out map.html
    bookmap web
    bookmap stats
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape, quoteattr

import typer
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from bookmap.config import DEFAULT_DB_PATH, GraphConfig, LayoutConfig, RecommendConfig
from bookmap.explain import format_explanation
from bookmap.identity import normalize_title
from bookmap.recommend import RecommendResult
from bookmap.recommend import recommend as run_recommend
from bookmap.store.db import Store

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.projection import GraphProjection

app = typer.Typer(
    name="bookmap",
    help="Build and query a book recommendation network.",
    no_args_is_help=True,
)

ingest_app = typer.Typer(help="Load a data source into the graph store.")
app.add_typer(ingest_app, name="ingest")

console = Console()

_BACKFILL_BATCH = 100_000
"""Rows per title_norm backfill batch. Large enough that the per-statement cost
disappears, small enough that 2.4M rows never sit in memory at once."""

_STAGING_VIEW = "_bookmap_title_norm"

_COMMUNITY_COLOURS = (
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
)
"""A fixed categorical palette, inlined because the map export must not fetch
anything. Community ids index into it modulo its length."""


# -- shared plumbing -------------------------------------------------------


def _fail(message: str) -> None:
    """Report a user-fixable problem and exit non-zero, without a traceback."""
    console.print(f"[bold red]error:[/bold red] {markup_escape(message)}")
    raise typer.Exit(1)


def _open_writable(db: str) -> Store:
    """Open (creating if absent) a store for writing."""
    try:
        return Store.open(db)
    except Exception as exc:  # noqa: BLE001 - duckdb raises a family of IO errors
        _fail(f"cannot open database {db}: {exc}")
        raise  # unreachable; keeps the return type honest


def _open_readable(db: str) -> Store:
    """Open a store read-only, as every query-side command should.

    Read-only both because DuckDB permits a single writer -- so a query can
    never collide with a running ingest -- and because a typo in ``--db`` should
    not silently create an empty database and report zero books.
    """
    path = Path(db)
    if not path.exists():
        _fail(f"no such database: {db} (run 'bookmap ingest demo' first)")
    try:
        return Store.open(path, read_only=True)
    except Exception as exc:  # noqa: BLE001 - duckdb raises a family of IO errors
        _fail(f"cannot open database {db}: {exc}")
        raise  # unreachable


def _source_or_fail(factory, *args: Any, **kwargs: Any):  # noqa: ANN001, ANN202
    """Construct a file-backed source, turning a bad path into a clean error.

    The adapters raise ``FileNotFoundError`` from their constructor; a dump path
    is the single most likely thing for a user to get wrong, and a traceback is
    not an error message.
    """
    try:
        return factory(*args, **kwargs)
    except FileNotFoundError as exc:
        _fail(str(exc))
        raise  # unreachable


def _split_seeds(seeds: str) -> list[str]:
    """Split the comma-separated ``--seeds`` value, dropping empties."""
    return [part.strip() for part in seeds.split(",") if part.strip()]


def explanation_titles(result: RecommendResult, store: Store) -> dict[str, str]:
    """Map every work id appearing in an explanation to its title.

    Seeds and recommendations carry their own titles, but the *waypoints* in the
    middle of a path are by definition neither, so they have to be fetched. They
    are also the part a reader most needs named: "2 hops from Dune via Dune
    Messiah" can be checked by eye, while a bare work id reads as a bug even when
    the path is right.

    Fetched in one batch rather than per node -- a full result set can walk a
    couple of hundred distinct books.
    """
    titles = {rec.work_id: rec.title for rec in result.recommendations}
    for match in result.seeds:
        if match.work_id and match.title:
            titles[match.work_id] = match.title

    waypoints = {
        work_id
        for rec in result.recommendations
        for explanation in rec.explanations
        for work_id in explanation.path
        if work_id not in titles
    }
    if waypoints:
        for work_id, book in store.get_books(waypoints).items():
            titles[work_id] = book.title
    # Anything still missing was pruned from the book table; showing the id beats
    # omitting the hop and silently shortening the path.
    for work_id in waypoints:
        titles.setdefault(work_id, work_id)
    return titles


def _counted(items: Iterable[Any], progress: Progress, task: int) -> Iterator[Any]:
    """Pass a stream through untouched while advancing a progress bar.

    Wrapping the iterator rather than pre-counting keeps ingest streaming: the
    9GB dump's row count is not known until it has been read.
    """
    count = 0
    for count, item in enumerate(items, start=1):  # noqa: B007 - count is used after the loop
        # Updated in blocks: refreshing per row costs more than the parse does.
        if count % 5_000 == 0:
            progress.update(task, completed=count)
        yield item
    progress.update(task, completed=count)


def backfill_title_norm(
    store: Store, *, batch_size: int = _BACKFILL_BATCH, quiet: bool = False
) -> int:
    """Populate ``books.title_norm`` wherever it is NULL. Returns rows updated.

    The bulk ``ingest_sql`` paths deliberately leave this column NULL rather
    than re-deriving :func:`bookmap.identity.normalize_title` in SQL, where it
    would drift from the Python definition. But ``title_norm`` is what backs
    title search, so until it is filled in every book from a real dump is
    invisible to the seed resolver -- the demo corpus works only because
    ``Store.upsert_books`` writes the column itself.

    Batched rather than one statement: 2.4M titles must cross into Python to be
    normalised, and the Arrow round trip per batch is what keeps that a
    bounded-memory scan instead of a 2.4M-row materialisation.
    """
    import pyarrow as pa

    conn = store.conn
    remaining = int(
        conn.execute("SELECT count(*) FROM books WHERE title_norm IS NULL").fetchone()[0]
    )
    if not remaining:
        return 0

    updated = 0
    with _backfill_progress(remaining, quiet=quiet) as (progress, task):
        while True:
            rows = conn.execute(
                "SELECT work_id, title FROM books WHERE title_norm IS NULL LIMIT ?",
                [batch_size],
            ).fetchall()
            if not rows:
                break

            batch = pa.table(
                {
                    "work_id": pa.array([row[0] for row in rows], type=pa.string()),
                    # normalize_title never returns NULL (the empty string is its
                    # answer for an unusable title), so every selected row is
                    # guaranteed to leave the WHERE clause and the loop advances.
                    "title_norm": pa.array(
                        [normalize_title(row[1] or "") for row in rows], type=pa.string()
                    ),
                }
            )
            conn.register(_STAGING_VIEW, batch)
            try:
                conn.execute(
                    f"UPDATE books SET title_norm = batch.title_norm "  # noqa: S608 - view name is a module constant
                    f"FROM {_STAGING_VIEW} AS batch WHERE books.work_id = batch.work_id"
                )
            finally:
                conn.unregister(_STAGING_VIEW)

            updated += len(rows)
            if progress is not None and task is not None:
                progress.update(task, completed=updated)
    return updated


class _NullProgress:
    """Stand-in used when progress output would corrupt machine-readable output."""

    def __enter__(self) -> tuple[None, None]:
        return None, None

    def __exit__(self, *exc: object) -> None:
        return None


def _backfill_progress(total: int, *, quiet: bool):  # noqa: ANN202
    if quiet:
        return _NullProgress()
    return _ProgressContext(total)


class _ProgressContext:
    def __init__(self, total: int) -> None:
        self._total = total
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

    def __enter__(self):  # noqa: ANN204
        self._progress.__enter__()
        task = self._progress.add_task("normalising titles", total=self._total)
        return self._progress, task

    def __exit__(self, *exc: object) -> None:
        self._progress.__exit__(*exc)


def _finish_ingest(store: Store, *, label: str) -> None:
    """The shared tail of every ingest: resolve refs, then fill ``title_norm``.

    Both steps have to happen after the books are in place -- refs resolve
    through the alias rows a book creates, and a title cannot be normalised
    before it is stored.
    """
    with console.status(f"resolving {label} edge endpoints…"):
        dropped = store.resolve_refs()
    filled = backfill_title_norm(store)

    counts = store.counts()
    console.print(
        f"[green]{label}[/green]: {counts['books']:,} books, "
        f"{counts['edges_raw']:,} raw edges"
        + (f", {dropped:,} unresolvable edges dropped" if dropped else "")
        + (f", {filled:,} titles normalised" if filled else "")
    )


# -- ingest ----------------------------------------------------------------


@ingest_app.command("demo")
def ingest_demo(db: str = DEFAULT_DB_PATH, path: str | None = None) -> None:
    """Load the bundled hand-authored demo corpus."""
    from bookmap.sources.demo import DemoSource

    source = _source_or_fail(DemoSource, path) if path else DemoSource()
    with _open_writable(db) as store:
        store.upsert_books(source.iter_books())
        store.insert_raw_edges(source.iter_edges())
        store.mark_ingest("demo", str(source.path), completed=True)
        _finish_ingest(store, label="demo")


@ingest_app.command("goodreads-ucsd")
def ingest_goodreads(
    path: str,
    db: str = DEFAULT_DB_PATH,
    max_rank: int = 50,
    authors: str | None = typer.Option(
        None,
        "--authors",
        help="Companion goodreads_book_authors.json(.gz); the books dump only "
        "references authors by id.",
    ),
) -> None:
    """Load goodreads_books.json.gz from the UCSD Book Graph."""
    from bookmap.sources.goodreads_ucsd import GoodreadsUCSDSource

    if authors is not None and not Path(authors).exists():
        _fail(f"no such authors file: {authors}")

    source = _source_or_fail(GoodreadsUCSDSource, path, authors)
    if authors is None:
        # Without it the dump's author_id references resolve to nothing, which
        # costs both display names and title+author seed matching.
        console.print(
            "[yellow]note:[/yellow] no --authors file given; books will be "
            "ingested without author names"
        )

    with _open_writable(db) as store:
        # Bulk SQL, not the streaming reader: the real dump is 9GB of NDJSON and
        # DuckDB's read_json scans it without it ever entering Python. That is
        # two statements, so a spinner is the only honest progress signal here.
        with console.status(f"reading {Path(path).name} (this takes a while)…"):
            books_added, edges_added = source.ingest_sql(store, max_rank=max_rank)
        console.print(f"read {books_added:,} books and {edges_added:,} edges")
        _finish_ingest(store, label="goodreads-ucsd")


@ingest_app.command("amazon-meta")
def ingest_amazon_meta(path: str, db: str = DEFAULT_DB_PATH, max_rank: int = 50) -> None:
    """Load SNAP amazon-meta.txt co-purchase data."""
    from bookmap.sources.amazon_meta import AmazonMetaSource

    source = _source_or_fail(AmazonMetaSource, path)
    with _open_writable(db) as store:
        _stream_into(store, source, label="amazon-meta", max_rank=max_rank)
        _finish_ingest(store, label="amazon-meta")


@ingest_app.command("amazon-reviews")
def ingest_amazon_reviews(path: str, db: str = DEFAULT_DB_PATH, max_rank: int = 50) -> None:
    """Load Amazon Reviews 2023 book metadata (also_buy / also_view)."""
    from bookmap.sources.amazon_meta import AmazonReviews2023Source

    source = _source_or_fail(AmazonReviews2023Source, path)
    with _open_writable(db) as store:
        with console.status(f"reading {Path(path).name}…"):
            books_added, edges_added = source.ingest_sql(store, max_rank=max_rank)
        console.print(f"read {books_added:,} books and {edges_added:,} edges")
        _finish_ingest(store, label="amazon-reviews")


@ingest_app.command("openlibrary")
def ingest_openlibrary(db: str = DEFAULT_DB_PATH, limit: int | None = None) -> None:
    """Enrich stored books with Open Library metadata."""
    import asyncio

    from bookmap.sources.openlibrary import OpenLibrarySource

    source = OpenLibrarySource()
    with _open_writable(db) as store:
        with console.status("querying Open Library…"):
            enriched = asyncio.run(source.enrich_store(store, limit=limit))
        console.print(f"[green]openlibrary[/green]: enriched {enriched:,} books")
        backfill_title_norm(store)


def _stream_into(store: Store, source: Any, *, label: str, max_rank: int) -> None:
    """Ingest through the Python readers, with a live row counter."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed:,} rows"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        books_task = progress.add_task(f"{label}: books", total=None)
        store.upsert_books(_counted(source.iter_books(), progress, books_task))
        edges_task = progress.add_task(f"{label}: edges", total=None)
        # max_rank is enforced at fusion time, so ranked rows are kept verbatim
        # here and the raw table stays retunable without a re-ingest.
        store.insert_raw_edges(_counted(source.iter_edges(), progress, edges_task))
    store.mark_ingest(label, str(getattr(source, "path", label)), completed=True)


# -- build -----------------------------------------------------------------


@app.command()
def build(
    db: str = DEFAULT_DB_PATH,
    min_weight: float = 0.05,
    drop_leaves: bool = False,
    max_rank: int = 50,
    communities: bool = True,
    betweenness: bool = False,
) -> None:
    """Fuse raw edges into the weighted graph and compute node metrics."""
    from bookmap.graph.build import fuse_in_sql, prune
    from bookmap.graph.centrality import approximate_betweenness
    from bookmap.graph.communities import community_sizes, detect_communities, label_communities
    from bookmap.store.projection import project

    config = GraphConfig(min_weight=min_weight, drop_leaves=drop_leaves, max_rank=max_rank)
    with _open_writable(db) as store:
        with console.status("fusing edges…"):
            # fuse_in_sql replaces edges_fused wholesale, which is what makes a
            # rebuild idempotent rather than cumulative -- edge weights are sums,
            # so a compounding build would silently inflate every one of them.
            fused = fuse_in_sql(store, config)
            if drop_leaves:
                # Leaf dropping is a whole-graph decision the SQL aggregate
                # cannot express, so that flag falls back to the Python pruner.
                fused = store.replace_fused_edges(
                    prune(store.iter_fused_edges(), min_weight=min_weight, drop_leaves=True)
                )
        if not fused:
            _fail("no fused edges: ingest a source before building")

        projection = project(store)
        console.print(f"fused graph: {projection.n_nodes:,} nodes, {projection.n_edges:,} edges")

        degree = projection.degree()
        weighted = projection.weighted_degree()

        labels = None
        if communities:
            with console.status("detecting communities…"):
                labels = detect_communities(projection)

        centrality = None
        if betweenness:
            # Off by default: even sampled, this is the most expensive thing in
            # the build, and nothing but the bridge-book view needs it.
            with console.status("sampling betweenness…"):
                centrality = approximate_betweenness(projection)

        # Cleared rather than upserted: a rebuild that produced fewer
        # communities, or dropped nodes, must not leave the old rows behind.
        store.conn.execute("DELETE FROM node_metrics")
        store.conn.execute("DELETE FROM communities")

        store.write_node_metrics(
            (
                projection.work_ids[row],
                int(degree[row]),
                float(weighted[row]),
                None if labels is None else int(labels[row]),
                None if centrality is None else float(centrality[row]),
            )
            for row in range(projection.n_nodes)
        )

        if labels is not None:
            titles, subjects = _aligned_metadata(store, projection)
            names = label_communities(labels, titles, subjects)
            sizes = community_sizes(labels)
            store.write_communities(
                (community, names.get(community, f"community {community}"), size)
                for community, size in sorted(sizes.items())
            )
            console.print(f"found {len(sizes)} communities")
            for community, size in sorted(sizes.items(), key=lambda item: -item[1])[:5]:
                label = names.get(community, "")
                console.print(f"  {community:>3}  {size:>6,}  {markup_escape(label)}")


def _aligned_metadata(
    store: Store, projection: GraphProjection
) -> tuple[list[str], list[tuple[str, ...]]]:
    """Titles and subjects in ``projection.work_ids`` order, for cluster naming.

    Streamed from the store rather than fetched by id list: TF-IDF labelling
    needs every member's vocabulary anyway, and one scan beats a millions-long
    IN clause.
    """
    metadata = {
        book.work_id: (book.title, book.subjects)
        for book in store.iter_books()
        if book.work_id in projection.index
    }
    titles = []
    subjects = []
    for work_id in projection.work_ids:
        title, subject = metadata.get(work_id, (work_id, ()))
        titles.append(title)
        subjects.append(subject)
    return titles, subjects


# -- recommend -------------------------------------------------------------


@app.command()
def recommend(
    seeds: str = typer.Option(..., help="Comma-separated book titles."),
    n: int = typer.Option(20, "-n", "--n", help="How many recommendations to return."),
    db: str = DEFAULT_DB_PATH,
    beta: float = 0.25,
    mmr_lambda: float = 0.7,
    exclude_same_author: bool = False,
    explain: bool = True,
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Recommend books similar to a set of seed titles."""
    from bookmap.store.projection import project

    queries = _split_seeds(seeds)
    if not queries:
        _fail("--seeds must name at least one book")

    config = RecommendConfig(
        hub_beta=beta,
        mmr_lambda=mmr_lambda,
        exclude_same_author=exclude_same_author,
    )
    with _open_readable(db) as store:
        projection = project(store)
        if projection.n_nodes == 0:
            _fail("the graph is empty: run 'bookmap build' first")
        result = run_recommend(queries, store, projection, n=n, config=config)

        if json_out:
            # Nothing else may reach stdout in this mode: the output is parsed.
            typer.echo(json.dumps(_result_payload(result), indent=2))
            return

        # Rendered inside the context: naming the books a path walks through
        # needs the store, since waypoints are neither seeds nor results.
        _print_result(result, store, explain=explain)

    if not result.recommendations and not result.seeds:
        raise typer.Exit(1)


def _result_payload(result: RecommendResult) -> dict[str, Any]:
    return {
        "seeds": [
            {
                "query": match.query,
                "work_id": match.work_id,
                "title": match.title,
                "authors": list(match.authors),
                "confidence": match.confidence,
                "alternatives": [list(pair) for pair in match.alternatives],
            }
            for match in result.seeds
        ],
        "unresolved": list(result.unresolved),
        "recommendations": [
            {
                "work_id": rec.work_id,
                "title": rec.title,
                "authors": list(rec.authors),
                "score": rec.score,
                "raw_ppr": rec.raw_ppr,
                "degree": rec.degree,
                "community": rec.community,
                "explanations": [
                    {
                        "seed_id": explanation.seed_id,
                        "seed_title": explanation.seed_title,
                        "path": list(explanation.path),
                        "hops": explanation.hops,
                        "strength": explanation.strength,
                    }
                    for explanation in rec.explanations
                ],
            }
            for rec in result.recommendations
        ],
    }


def _print_result(result: RecommendResult, store: Store, *, explain: bool) -> None:
    for match in result.seeds:
        label = f"{match.title} — {', '.join(match.authors)}" if match.authors else match.title
        console.print(f"[dim]seed[/dim] {markup_escape(str(label))}")
        if match.ambiguous:
            others = "; ".join(name for _, name in match.alternatives[1:])
            console.print(f"      [yellow]also matched:[/yellow] {markup_escape(others)}")

    if result.unresolved:
        # Surfaced, never silently dropped: a seed the user thought they gave
        # you changes the answer by its absence.
        console.print(
            "[yellow]no match for:[/yellow] "
            + markup_escape(", ".join(result.unresolved))
        )

    if not result.recommendations:
        console.print("[yellow]no recommendations[/yellow]")
        return

    console.print()
    # Printed as blocks rather than a table so a long title wraps instead of
    # being truncated, and so each explanation sits with its recommendation.
    titles = explanation_titles(result, store)

    for position, rec in enumerate(result.recommendations, start=1):
        author = ", ".join(rec.authors) or "unknown"
        console.print(
            f"[bold]{position:>3}[/bold]  {markup_escape(rec.title)} "
            f"[dim]—[/dim] {markup_escape(author)}  [cyan]{rec.score:.4f}[/cyan]"
        )
        if explain:
            for explanation in rec.explanations:
                console.print(
                    f"      [dim]{markup_escape(format_explanation(explanation, titles))}[/dim]"
                )


# -- map -------------------------------------------------------------------


@app.command("map")
def map_cmd(
    seeds: str = typer.Option(..., help="Comma-separated book titles."),
    out: str = "map.html",
    n: int = 40,
    db: str = DEFAULT_DB_PATH,
) -> None:
    """Export a standalone HTML map of the seeds' neighbourhood."""
    from bookmap.graph.layout import force_atlas2, normalize_positions
    from bookmap.store.projection import project

    queries = _split_seeds(seeds)
    if not queries:
        _fail("--seeds must name at least one book")

    layout_config = LayoutConfig()
    with _open_readable(db) as store:
        projection = project(store)
        if projection.n_nodes == 0:
            _fail("the graph is empty: run 'bookmap build' first")

        result = run_recommend(queries, store, projection, n=n)
        seed_ids = [match.work_id for match in result.seeds if match.work_id]
        member_ids = _map_members(projection, seed_ids, result, layout_config.max_nodes)
        if not member_ids:
            _fail("nothing to draw: no seed resolved onto the graph")

        subgraph = projection.subgraph(member_ids)
        positions = normalize_positions(force_atlas2(subgraph, layout_config))
        books = store.get_books(subgraph.work_ids)
        communities = {
            work_id: store.node_community(work_id) for work_id in subgraph.work_ids
        }
        labels = store.community_labels()

    html = _render_map(
        subgraph,
        positions,
        books,
        communities,
        labels,
        seed_ids=set(seed_ids),
        queries=queries,
    )
    Path(out).write_text(html, encoding="utf-8")
    console.print(f"wrote [green]{out}[/green] ({subgraph.n_nodes} nodes, {subgraph.n_edges} edges)")


def _map_members(
    projection: GraphProjection,
    seed_ids: list[str],
    result: RecommendResult,
    max_nodes: int,
) -> list[str]:
    """Nodes to draw: the seeds, the results, then their shared neighbours.

    Neighbours are included because a map of only the answer shows no context --
    the interesting part is the tissue connecting the seeds to the results --
    but the layout is quadratic in node count, so the list is capped.
    """
    members: dict[str, None] = {}
    for work_id in seed_ids:
        if work_id in projection.index:
            members[work_id] = None
    for rec in result.recommendations:
        members.setdefault(rec.work_id, None)
    # Explanation waypoints first: they are the edges the user was *told* about,
    # so a map missing them would contradict the recommendation list.
    for rec in result.recommendations:
        for explanation in rec.explanations:
            for work_id in explanation.path:
                if len(members) >= max_nodes:
                    break
                members.setdefault(work_id, None)

    for work_id in list(members):
        if len(members) >= max_nodes:
            break
        rows, _ = projection.neighbors_of(work_id)
        for row in rows.tolist():
            if len(members) >= max_nodes:
                break
            members.setdefault(projection.work_ids[row], None)
    return list(members)


def _render_map(
    subgraph: GraphProjection,
    positions: Any,
    books: dict[str, Any],
    communities: dict[str, int | None],
    labels: dict[int, str],
    *,
    seed_ids: set[str],
    queries: list[str],
) -> str:
    """Render the subgraph as one self-contained HTML file.

    Inline SVG with no scripts, styles, fonts or images fetched from anywhere:
    the export has to open from a thumb drive on a plane. The interactive
    version of this view is the web app; this is the artefact you can email.
    """
    import numpy as np
    import scipy.sparse as sp

    width = height = 1000.0
    degree = subgraph.degree()
    max_degree = max(float(degree.max()) if degree.size else 1.0, 1.0)

    parts = [
        "<title>bookmap — " + escape(", ".join(queries)) + "</title>",
        "<style>",
        "  :root { color-scheme: light dark; }",
        "  body { margin: 0; font: 14px system-ui, sans-serif; background: #fbfbfa; color: #1a1a1a; }",
        "  @media (prefers-color-scheme: dark) { body { background: #16181d; color: #e8e8e8; } }",
        "  header { padding: 1rem 1.25rem; }",
        "  h1 { font-size: 1.1rem; margin: 0 0 .25rem; font-weight: 600; }",
        "  .legend { display: flex; flex-wrap: wrap; gap: .75rem; padding: 0 1.25rem 1rem; font-size: 12px; }",
        "  .swatch { width: .7rem; height: .7rem; border-radius: 50%; display: inline-block; margin-right: .3rem; }",
        "  .wrap { overflow-x: auto; padding: 0 1.25rem 2rem; }",
        "  svg { max-width: 100%; height: auto; display: block; }",
        "  text { font: 11px system-ui, sans-serif; }",
        "  .seed { font-weight: 700; }",
        "</style>",
        "<header>",
        f"  <h1>bookmap · {escape(', '.join(queries))}</h1>",
        f"  <div>{subgraph.n_nodes} books, {subgraph.n_edges} connections. "
        "Larger circles are better-connected books; bold labels are your seeds.</div>",
        "</header>",
    ]

    present = sorted({c for c in communities.values() if c is not None})
    if present:
        parts.append('<div class="legend">')
        for community in present:
            colour = _COMMUNITY_COLOURS[community % len(_COMMUNITY_COLOURS)]
            label = labels.get(community) or f"community {community}"
            parts.append(
                f'  <span><span class="swatch" style="background:{colour}"></span>'
                f"{escape(label)}</span>"
            )
        parts.append("</div>")

    parts.append('<div class="wrap">')
    parts.append(
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" '
        'xmlns="http://www.w3.org/2000/svg" role="img">'
    )

    # Upper triangle only: the adjacency is symmetric and drawing both copies
    # would double every line's opacity.
    upper = sp.triu(sp.csr_matrix(subgraph.adjacency), k=1).tocoo()
    weights = upper.data.astype(float)
    strongest = float(weights.max()) if weights.size else 1.0
    parts.append('<g stroke="#8a8f98">')
    for i, j, weight in zip(upper.row.tolist(), upper.col.tolist(), weights.tolist()):
        opacity = 0.12 + 0.5 * (weight / strongest if strongest else 0.0)
        parts.append(
            f'<line x1="{positions[i][0]:.1f}" y1="{positions[i][1]:.1f}"'
            f' x2="{positions[j][0]:.1f}" y2="{positions[j][1]:.1f}"'
            f' stroke-opacity="{opacity:.2f}" stroke-width="{0.4 + 1.6 * weight / strongest:.2f}"/>'
        )
    parts.append("</g>")

    order = np.argsort(degree)  # draw hubs last so they are not buried
    for row in order.tolist():
        work_id = subgraph.work_ids[row]
        book = books.get(work_id)
        title = book.title if book is not None else work_id
        community = communities.get(work_id)
        colour = (
            _COMMUNITY_COLOURS[community % len(_COMMUNITY_COLOURS)]
            if community is not None
            else "#8a8f98"
        )
        radius = 4.0 + 9.0 * (float(degree[row]) / max_degree) ** 0.5
        is_seed = work_id in seed_ids
        x, y = float(positions[row][0]), float(positions[row][1])

        tooltip = title if book is None else f"{title} — {', '.join(book.authors) or 'unknown'}"
        node = (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{colour}"'
            f' stroke="{"#1a1a1a" if is_seed else colour}"'
            f' stroke-width="{2.0 if is_seed else 0.0}" fill-opacity="0.85">'
            f"<title>{escape(tooltip)}</title></circle>"
        )
        # The one outbound link the export is allowed: a Goodreads id is the
        # only stable public address a book node has.
        goodreads_id = getattr(book, "goodreads_id", None) if book is not None else None
        if goodreads_id:
            href = quoteattr(f"https://www.goodreads.com/book/show/{goodreads_id}")
            node = f"<a href={href} target=_blank>{node}</a>"
        parts.append(node)

        # Only label what a reader can actually take in: seeds always, and the
        # better-connected nodes otherwise.
        if is_seed or float(degree[row]) > max_degree * 0.35:
            classes = ' class="seed"' if is_seed else ""
            parts.append(
                f'<text x="{x + radius + 3:.1f}" y="{y + 3:.1f}"{classes}'
                f' fill="currentColor">{escape(title[:42])}</text>'
            )

    parts.append("</svg></div>")
    return "\n".join(parts) + "\n"


# -- web / stats -----------------------------------------------------------


@app.command()
def web(db: str = DEFAULT_DB_PATH, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the interactive map."""
    import uvicorn

    from bookmap.web.app import create_app

    if not Path(db).exists():
        _fail(f"no such database: {db} (run 'bookmap ingest demo' first)")
    console.print(f"serving [green]http://{host}:{port}[/green] from {db}")
    uvicorn.run(create_app(db), host=host, port=port)


@app.command()
def stats(db: str = DEFAULT_DB_PATH) -> None:
    """Print graph size, degree distribution, and community summary."""
    import numpy as np

    from bookmap.store.projection import project

    with _open_readable(db) as store:
        counts = store.counts()
        table = Table(title="store", show_header=True, header_style="bold")
        table.add_column("table")
        table.add_column("rows", justify="right")
        for name, value in counts.items():
            table.add_row(name, f"{value:,}")
        console.print(table)

        if not counts["edges_fused"]:
            console.print("[yellow]no fused graph yet: run 'bookmap build'[/yellow]")
            return

        projection = project(store)
        degree = projection.degree()
        console.print(
            f"\ngraph: {projection.n_nodes:,} nodes, {projection.n_edges:,} edges"
        )
        console.print(
            "degree: "
            f"mean {degree.mean():.1f}, median {np.median(degree):.0f}, "
            f"p95 {np.percentile(degree, 95):.0f}, max {degree.max():.0f}"
        )

        rows = store.conn.execute(
            "SELECT community, label, size FROM communities ORDER BY size DESC LIMIT 10"
        ).fetchall()
        if rows:
            community_table = Table(title="largest communities", header_style="bold")
            community_table.add_column("id", justify="right")
            community_table.add_column("size", justify="right")
            community_table.add_column("label")
            for community, label, size in rows:
                community_table.add_row(str(community), f"{size:,}", label or "")
            console.print(community_table)


if __name__ == "__main__":  # pragma: no cover
    app()
