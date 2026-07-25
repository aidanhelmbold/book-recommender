"""Bridge books: the titles that join two otherwise-separate reading communities.

Shared by the CLI and the web API. It lives here rather than in either of them
because the two fiddly parts -- reading communities back rather than recomputing
them, and over-fetching before filtering to seeded clusters -- are easy to get
subtly wrong twice, and a divergence would show up as the two surfaces disagreeing
about what a bridge is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from bookmap.graph.centrality import bridge_books
from bookmap.recommend import resolve_seeds

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.db import Store
    from bookmap.store.projection import GraphProjection

OVERFETCH = 8
"""How much deeper than ``top_n`` to search when filtering to seeded clusters.

The globally strongest bridges frequently touch none of the seeded communities, so
trimming after the fact would quietly return fewer results than asked for.
"""

NO_COMMUNITY = -1
"""Marker for a node ``build`` never assigned a community.

Kept distinct from every real id so such nodes cannot be mistaken for members of
one, and dropped before scoring -- a cluster of "unassigned" is not a genre, and
counting it would invent bridges to nowhere.
"""


class BridgesUnavailable(RuntimeError):
    """Raised when the graph has no communities to bridge between."""


def stored_communities(store: Store, projection: GraphProjection) -> np.ndarray:
    """Community id per projection row, read from ``node_metrics`` in one query.

    Read rather than recomputed: detection on a 2.4M-node graph is minutes of work
    that ``build`` has already done, and recomputing here could disagree with the
    ids every other surface reports.
    """
    rows = store.conn.execute(
        "SELECT work_id, community FROM node_metrics WHERE community IS NOT NULL"
    ).fetchall()
    if not rows:
        raise BridgesUnavailable(
            "no communities in this database — run 'bookmap build' first "
            "(bridges are defined between clusters, so there is nothing to report "
            "until they exist)"
        )
    by_work = dict(rows)
    return np.fromiter(
        (by_work.get(work_id, NO_COMMUNITY) for work_id in projection.work_ids),
        dtype=np.int64,
        count=projection.n_nodes,
    )


def find_bridges(
    store: Store,
    projection: GraphProjection,
    *,
    seeds: list[str] | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    """Rank bridge books, optionally restricted to the seeds' communities.

    With seeds given, the useful answer is not the strongest bridge anywhere in the
    graph but the books joining the genres the reader already reads.
    """
    if projection.n_nodes == 0:
        raise BridgesUnavailable("the graph is empty: run 'bookmap build' first")

    labels = stored_communities(store, projection)
    community_names = store.community_labels()

    seed_communities: list[int] = []
    unresolved: list[str] = []
    seed_titles: list[str] = []
    if seeds:
        matches, unresolved = resolve_seeds(seeds, store)
        seed_titles = [match.title for match in matches if match.title]
        seed_communities = sorted(
            {
                community
                for match in matches
                if match.work_id is not None
                and (community := store.node_community(match.work_id)) is not None
            }
        )

    wanted = top_n * OVERFETCH if seed_communities else top_n
    ranked = bridge_books(projection, labels, top_n=wanted)

    found: list[tuple[str, float, tuple[int, ...]]] = []
    for work_id, score, communities in ranked:
        touched = tuple(c for c in communities if c != NO_COMMUNITY)
        # Touching one community is not bridging anything.
        if len(touched) < 2:
            continue
        if seed_communities and not set(touched) & set(seed_communities):
            continue
        found.append((work_id, score, touched))
        if len(found) >= top_n:
            break

    books = store.get_books([work_id for work_id, _, _ in found])
    return {
        "seed_communities": seed_communities,
        "seed_titles": seed_titles,
        "unresolved": unresolved,
        "bridges": [
            {
                "work_id": work_id,
                "title": books[work_id].title if work_id in books else work_id,
                "authors": list(books[work_id].authors) if work_id in books else [],
                "score": score,
                "communities": list(communities),
                "community_labels": [
                    community_names.get(community, "") for community in communities
                ],
            }
            for work_id, score, communities in found
        ],
    }
