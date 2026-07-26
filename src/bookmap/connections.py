"""How a set of books connects: seed resolution, the skeleton, and titles.

Shared by the CLI and the web API, for the same reason ``bridges.py`` is: the
fiddly parts -- resolving typed titles onto works, and naming every node on the
route rather than emitting work ids -- would be easy to get subtly wrong twice,
and a divergence would show up as the two surfaces disagreeing about how two
books connect.

``graph/connect.py`` answers the graph question and knows nothing about storage.
This is the layer that turns its work ids back into books.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bookmap.graph.connect import DEFAULT_MAX_HOPS, Connections, connect_seeds
from bookmap.recommend import resolve_seeds

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.db import Store
    from bookmap.store.projection import GraphProjection


class ConnectionsUnavailable(RuntimeError):
    """Raised when there is no graph to route over."""


def find_connections(
    store: Store,
    projection: GraphProjection,
    *,
    seeds: list[str],
    max_hops: int = DEFAULT_MAX_HOPS,
) -> dict[str, Any]:
    """Resolve ``seeds``, find the skeleton joining them, and name every node.

    Titles ride alongside ids throughout: the ids are what a caller re-queries
    with and the titles are what it displays, so dropping either would force a
    second round trip for every route.
    """
    if projection.n_nodes == 0:
        raise ConnectionsUnavailable("the graph is empty: run 'bookmap build' first")

    matches, unresolved = resolve_seeds(seeds, store)
    # Resolution order is the order the user typed, and the skeleton preserves it,
    # so the report reads back in the same order.
    terminals = [match.work_id for match in matches if match.work_id is not None]
    result = connect_seeds(projection, terminals, max_hops=max_hops)
    titles = _titles(store, result, matches)

    return {
        "seeds": [
            {
                "query": match.query,
                "work_id": match.work_id,
                "title": match.title,
                "authors": list(match.authors),
            }
            for match in matches
        ],
        "unresolved": list(unresolved),
        "max_hops": max_hops,
        "skeletons": [_skeleton_payload(skeleton, titles) for skeleton in result.skeletons],
        # Present seeds that reached no other seed. Named explicitly because a
        # partial answer presented as a whole one is the failure mode here.
        "unjoined": [
            {"work_id": work_id, "title": titles.get(work_id, work_id)}
            for work_id in result.unjoined
        ],
        "capped": [
            {
                "source": pair.source,
                "source_title": titles.get(pair.source, pair.source),
                "target": pair.target,
                "target_title": titles.get(pair.target, pair.target),
                "hops": pair.hops,
                "limit": pair.limit,
            }
            for pair in result.capped
        ],
    }


def _skeleton_payload(skeleton: Any, titles: dict[str, str]) -> dict[str, Any]:
    def named(work_ids: tuple[str, ...]) -> list[dict[str, str]]:
        return [
            {"work_id": work_id, "title": titles.get(work_id, work_id)}
            for work_id in work_ids
        ]

    return {
        "terminals": named(skeleton.terminals),
        "connectors": named(skeleton.connectors),
        "nodes": named(skeleton.nodes),
        "edges": [
            {"source": src, "target": dst, "weight": weight}
            for src, dst, weight in skeleton.edges
        ],
        "cost": skeleton.cost,
        "strength": skeleton.strength,
        "legs": [
            {
                "source": leg.source,
                "target": leg.target,
                "path": list(leg.path),
                "titles": [titles.get(work_id, work_id) for work_id in leg.path],
                "hops": leg.hops,
                "cost": leg.cost,
                "strength": leg.strength,
            }
            for leg in skeleton.legs
        ],
    }


def _titles(store: Store, result: Connections, matches: list[Any]) -> dict[str, str]:
    """Title for every work id the answer mentions, fetched in one batch.

    The connectors are the whole point of the feature and are by definition not
    seeds, so they have to be looked up. A bare work id in the route reads as a
    bug even when the route is right.
    """
    titles = {
        match.work_id: match.title
        for match in matches
        if match.work_id is not None and match.title
    }
    wanted = {
        work_id
        for skeleton in result.skeletons
        for leg in skeleton.legs
        for work_id in leg.path
        if work_id not in titles
    }
    wanted |= {work_id for work_id in result.nodes if work_id not in titles}
    if wanted:
        for work_id, book in store.get_books(wanted).items():
            titles[work_id] = book.title
    # Anything still missing was pruned from the book table; showing the id beats
    # omitting the hop and silently shortening the route.
    for work_id in wanted:
        titles.setdefault(work_id, work_id)
    return titles
