"""``find_connections`` must not drop a seed on the floor.

On the real graph only ~26% of books carry a fused edge (392k nodes out of 1.52M
works), so "this title resolved to a real book that has no node in the graph" is
the *common* case, not an exotic one. It is also invisible: the seed resolves, it
is echoed back to the user, and then it silently plays no part in the answer.

This is the failure mode the whole design is meant to rule out — a partial answer
presented as a whole one — and it slipped through because every book in the demo
corpus has edges, so no existing test could reach the branch.
"""

from __future__ import annotations

import pytest

from bookmap.connections import find_connections
from bookmap.graph.build import fuse_edges
from bookmap.models import Book, Edge, EdgeKind, SourceName


@pytest.fixture
def store_with_an_isolated_book(store):
    """Two joined books plus one that resolves but has no fused edge."""
    books = [
        Book(work_id="w-a", title="Alpha", authors=("A",)),
        Book(work_id="w-mid", title="Middle", authors=("M",)),
        Book(work_id="w-b", title="Beta", authors=("B",)),
        # In the books table, reachable by title search, absent from the graph.
        Book(work_id="w-lonely", title="Lonely Volume", authors=("L",)),
    ]
    edges = []
    for a, b in (("w-a", "w-mid"), ("w-mid", "w-b")):
        edges.append(Edge(a, b, EdgeKind.GR_SIMILAR, 1, SourceName.DEMO))
        edges.append(Edge(b, a, EdgeKind.GR_SIMILAR, 1, SourceName.DEMO))

    store.upsert_books(books)
    store.insert_raw_edges(edges)
    store.replace_fused_edges(fuse_edges(edges))
    return store


@pytest.fixture
def isolated_projection(store_with_an_isolated_book):
    from bookmap.store.projection import project

    return project(store_with_an_isolated_book)


class TestSeedsAbsentFromTheGraph:
    def test_a_resolved_but_edgeless_seed_is_reported(
        self, store_with_an_isolated_book, isolated_projection
    ) -> None:
        payload = find_connections(
            store_with_an_isolated_book,
            isolated_projection,
            seeds=["Alpha", "Beta", "Lonely Volume"],
        )
        # It resolved, so it is not "unresolved" -- a different fact needing a
        # different word, or the user goes looking for a typo they did not make.
        assert payload["unresolved"] == []
        assert [entry["title"] for entry in payload["missing"]] == ["Lonely Volume"]

    def test_the_reachable_seeds_are_still_joined(
        self, store_with_an_isolated_book, isolated_projection
    ) -> None:
        payload = find_connections(
            store_with_an_isolated_book,
            isolated_projection,
            seeds=["Alpha", "Beta", "Lonely Volume"],
        )
        assert payload["skeletons"], "one bad seed must not sink the answer"
        connectors = [
            node["title"] for node in payload["skeletons"][0]["connectors"]
        ]
        assert connectors == ["Middle"]

    def test_missing_carries_a_title_not_a_work_id(
        self, store_with_an_isolated_book, isolated_projection
    ) -> None:
        payload = find_connections(
            store_with_an_isolated_book,
            isolated_projection,
            seeds=["Alpha", "Beta", "Lonely Volume"],
        )
        entry = payload["missing"][0]
        assert entry["work_id"] == "w-lonely"
        assert entry["title"] == "Lonely Volume"

    def test_nothing_missing_when_every_seed_is_in_the_graph(
        self, store_with_an_isolated_book, isolated_projection
    ) -> None:
        payload = find_connections(
            store_with_an_isolated_book, isolated_projection, seeds=["Alpha", "Beta"]
        )
        assert payload["missing"] == []
