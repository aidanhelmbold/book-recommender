"""DuckDB store behaviour.

The idempotency tests here guard a specific, quiet bug class: fusion sums over
``edges_raw``, so any path that lets a re-ingest duplicate rows silently
inflates edge weights and skews every recommendation afterwards. That failure
would not raise, and would not be visible in a spot check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bookmap.models import Book, Edge, EdgeKind, FusedEdge, SourceName
from bookmap.store.db import Store


class TestLifecycle:
    def test_open_creates_file_and_schema(self, db_path: Path) -> None:
        with Store.open(db_path) as store:
            assert store.counts()["books"] == 0
        assert db_path.exists()

    def test_apply_schema_is_idempotent(self, store: Store) -> None:
        store.apply_schema()
        store.apply_schema()
        assert store.counts()["books"] == 0

    def test_reopen_preserves_data(self, db_path: Path, sample_books: list[Book]) -> None:
        with Store.open(db_path) as store:
            store.upsert_books(sample_books)
        with Store.open(db_path) as store:
            assert store.counts()["books"] == len(sample_books)

    def test_read_only_rejects_writes(self, db_path: Path, sample_books: list[Book]) -> None:
        """The web app opens read-only so serving cannot collide with an ingest."""
        with Store.open(db_path) as store:
            store.upsert_books(sample_books)
        with Store.open(db_path, read_only=True) as store:
            assert store.counts()["books"] == len(sample_books)
            with pytest.raises(Exception):  # noqa: B017 - duckdb raises its own type
                store.upsert_books([Book(work_id="w-new", title="New")])


class TestBooks:
    def test_upsert_and_get(self, store: Store, sample_books: list[Book]) -> None:
        assert store.upsert_books(sample_books) == len(sample_books)
        got = store.get_book("w-dune")
        assert got is not None
        assert got.title == "Dune"
        assert got.authors == ("Frank Herbert",)
        assert got.year == 1965

    def test_upsert_is_idempotent(self, store: Store, sample_books: list[Book]) -> None:
        store.upsert_books(sample_books)
        store.upsert_books(sample_books)
        assert store.counts()["books"] == len(sample_books)

    def test_upsert_updates_rather_than_duplicates(self, store: Store) -> None:
        store.upsert_books([Book(work_id="w-x", title="Dune", year=None)])
        store.upsert_books([Book(work_id="w-x", title="Dune", year=1965)])
        assert store.counts()["books"] == 1
        book = store.get_book("w-x")
        assert book is not None
        assert book.year == 1965

    def test_get_missing_book_returns_none(self, store: Store) -> None:
        assert store.get_book("nope") is None

    def test_get_books_batch_skips_missing(self, store: Store, sample_books: list[Book]) -> None:
        store.upsert_books(sample_books)
        got = store.get_books(["w-dune", "w-foundation", "nope"])
        assert set(got) == {"w-dune", "w-foundation"}

    def test_iter_books_round_trips(self, store: Store, sample_books: list[Book]) -> None:
        store.upsert_books(sample_books)
        titles = {b.title for b in store.iter_books()}
        assert titles == {b.title for b in sample_books}

    def test_upsert_populates_aliases(self, store: Store, sample_books: list[Book]) -> None:
        store.upsert_books(sample_books)
        assert store.alias_lookup("goodreads_id", "234225") == "w-dune"
        assert store.alias_lookup("isbn13", "9780441013593") == "w-dune"

    def test_search_titles(self, store: Store, sample_books: list[Book]) -> None:
        store.upsert_books(sample_books)
        hits = store.search_titles("dune")
        assert {b.work_id for b in hits} >= {"w-dune", "w-dune-messiah"}

    def test_search_titles_respects_limit(self, store: Store, sample_books: list[Book]) -> None:
        store.upsert_books(sample_books)
        assert len(store.search_titles("dune", limit=1)) == 1

    def test_title_candidates_shape(self, store: Store, sample_books: list[Book]) -> None:
        store.upsert_books(sample_books)
        candidates = list(store.title_candidates())
        assert len(candidates) == len(sample_books)
        work_id, title, authors = candidates[0]
        assert isinstance(work_id, str)
        assert isinstance(title, str)
        assert isinstance(authors, tuple)


class TestRawEdges:
    def test_insert_and_iterate(self, store: Store, sample_books, sample_edges) -> None:
        store.upsert_books(sample_books)
        assert store.insert_raw_edges(sample_edges) == len(sample_edges)
        assert store.counts()["edges_raw"] == len(sample_edges)

    def test_insert_is_idempotent(self, store: Store, sample_books, sample_edges) -> None:
        """Re-ingesting a dump must not accumulate rows.

        Fusion sums over this table, so duplicates would double edge weights
        without any visible error.
        """
        store.upsert_books(sample_books)
        store.insert_raw_edges(sample_edges)
        store.insert_raw_edges(sample_edges)
        assert store.counts()["edges_raw"] == len(sample_edges)

    def test_same_pair_different_kind_coexists(self, store: Store) -> None:
        """A pair attested by both Goodreads and Amazon is two rows, not one."""
        edges = [
            Edge("a", "b", EdgeKind.GR_SIMILAR, 1, SourceName.GOODREADS_UCSD),
            Edge("a", "b", EdgeKind.AZ_ALSO_BOUGHT, 1, SourceName.AMAZON_META),
        ]
        store.insert_raw_edges(edges)
        assert store.counts()["edges_raw"] == 2

    def test_rank_is_preserved(self, store: Store) -> None:
        store.insert_raw_edges(
            [Edge("a", "b", EdgeKind.GR_SIMILAR, 7, SourceName.GOODREADS_UCSD)]
        )
        edge = next(iter(store.iter_raw_edges()))
        assert edge.rank == 7
        assert edge.kind == EdgeKind.GR_SIMILAR


class TestRefResolution:
    def test_namespaced_refs_are_rewritten_to_work_ids(self, store: Store) -> None:
        """Adapters emit ``source:raw_id``; those must become canonical ids.

        A dump routinely references a book before describing it, which is why
        this is a separate pass rather than done during streaming.
        """
        store.upsert_books(
            [
                Book(work_id="w-dune", title="Dune", goodreads_id="234225"),
                Book(work_id="w-messiah", title="Dune Messiah", goodreads_id="106"),
            ]
        )
        store.insert_raw_edges(
            [
                Edge(
                    "goodreads_ucsd:234225",
                    "goodreads_ucsd:106",
                    EdgeKind.GR_SIMILAR,
                    1,
                    SourceName.GOODREADS_UCSD,
                )
            ]
        )
        store.resolve_refs()
        edges = list(store.iter_raw_edges())
        assert len(edges) == 1
        assert (edges[0].src, edges[0].dst) == ("w-dune", "w-messiah")

    def test_unresolvable_edges_are_dropped(self, store: Store) -> None:
        """A "similar" pointer to a book absent from the dump is not a node.

        The Goodreads dump references books outside its own corpus; keeping those
        would create titleless phantom nodes in the map.
        """
        store.upsert_books([Book(work_id="w-dune", title="Dune", goodreads_id="234225")])
        store.insert_raw_edges(
            [
                Edge(
                    "goodreads_ucsd:234225",
                    "goodreads_ucsd:000000",
                    EdgeKind.GR_SIMILAR,
                    1,
                    SourceName.GOODREADS_UCSD,
                )
            ]
        )
        dropped = store.resolve_refs()
        assert dropped == 1
        assert store.counts()["edges_raw"] == 0

    def test_resolve_refs_is_idempotent(self, store: Store) -> None:
        store.upsert_books(
            [
                Book(work_id="w-dune", title="Dune", goodreads_id="234225"),
                Book(work_id="w-messiah", title="Dune Messiah", goodreads_id="106"),
            ]
        )
        store.insert_raw_edges(
            [
                Edge(
                    "goodreads_ucsd:234225",
                    "goodreads_ucsd:106",
                    EdgeKind.GR_SIMILAR,
                    1,
                    SourceName.GOODREADS_UCSD,
                )
            ]
        )
        store.resolve_refs()
        store.resolve_refs()
        assert store.counts()["edges_raw"] == 1

    def test_asin_refs_resolve_through_the_asin_alias(self, store: Store) -> None:
        store.upsert_books(
            [
                Book(work_id="w-a", title="A", asin="B000FC1PWA"),
                Book(work_id="w-b", title="B", asin="B000FC1PWB"),
            ]
        )
        store.insert_raw_edges(
            [
                Edge(
                    "amazon_meta:B000FC1PWA",
                    "amazon_meta:B000FC1PWB",
                    EdgeKind.AZ_ALSO_BOUGHT,
                    1,
                    SourceName.AMAZON_META,
                )
            ]
        )
        store.resolve_refs()
        edges = list(store.iter_raw_edges())
        assert (edges[0].src, edges[0].dst) == ("w-a", "w-b")


class TestFusedEdges:
    def test_replace_writes_the_graph(self, store: Store) -> None:
        assert store.replace_fused_edges([FusedEdge("a", "b", 0.5)]) == 1
        assert store.counts()["edges_fused"] == 1

    def test_replace_is_not_additive(self, store: Store) -> None:
        """Rebuilding the graph replaces it; weights must not compound."""
        store.replace_fused_edges([FusedEdge("a", "b", 0.5)])
        store.replace_fused_edges([FusedEdge("a", "b", 0.5)])
        assert store.counts()["edges_fused"] == 1
        edge = next(iter(store.iter_fused_edges()))
        assert edge.weight == pytest.approx(0.5)

    def test_neighbors_are_strongest_first(self, store: Store) -> None:
        store.replace_fused_edges(
            [
                FusedEdge("a", "b", 0.2),
                FusedEdge("a", "c", 0.9),
                FusedEdge("a", "d", 0.5),
            ]
        )
        neighbors = store.neighbors("a")
        assert [n for n, _ in neighbors] == ["c", "d", "b"]

    def test_neighbors_are_symmetric(self, store: Store) -> None:
        """Fused edges are stored once with src < dst but must resolve both ways."""
        store.replace_fused_edges([FusedEdge("a", "z", 0.7)])
        assert store.neighbors("z") == [("a", pytest.approx(0.7))]

    def test_neighbors_respects_limit(self, store: Store) -> None:
        store.replace_fused_edges([FusedEdge("a", f"n{i}", 0.5) for i in range(10)])
        assert len(store.neighbors("a", limit=3)) == 3

    def test_neighbors_of_unknown_node(self, store: Store) -> None:
        assert store.neighbors("nope") == []


class TestMetrics:
    def test_write_and_read_node_metrics(self, store: Store) -> None:
        store.write_node_metrics([("w-dune", 4, 2.5, 1, 0.3)])
        assert store.node_community("w-dune") == 1

    def test_node_metrics_upsert(self, store: Store) -> None:
        store.write_node_metrics([("w-dune", 4, 2.5, 1, 0.3)])
        store.write_node_metrics([("w-dune", 5, 3.0, 2, 0.4)])
        assert store.counts()["node_metrics"] == 1
        assert store.node_community("w-dune") == 2

    def test_community_labels(self, store: Store) -> None:
        store.write_communities([(0, "hard science fiction", 42), (1, "literary fiction", 30)])
        labels = store.community_labels()
        assert labels[0] == "hard science fiction"
        assert labels[1] == "literary fiction"

    def test_node_community_unknown(self, store: Store) -> None:
        assert store.node_community("nope") is None


class TestIngestState:
    def test_records_progress(self, store: Store) -> None:
        store.mark_ingest("goodreads_ucsd", "/tmp/x.gz", rows_done=1000, books_added=900)
        assert store.counts()["ingest_state"] == 1

    def test_updates_in_place(self, store: Store) -> None:
        """Resumability depends on one row per (source, artifact)."""
        store.mark_ingest("goodreads_ucsd", "/tmp/x.gz", rows_done=1000)
        store.mark_ingest("goodreads_ucsd", "/tmp/x.gz", rows_done=2000, completed=True)
        assert store.counts()["ingest_state"] == 1


class TestCounts:
    def test_reports_every_table(self, store: Store) -> None:
        counts = store.counts()
        for table in (
            "books",
            "edges_raw",
            "edges_fused",
            "aliases",
            "node_metrics",
            "communities",
            "ingest_state",
        ):
            assert table in counts
            assert counts[table] == 0
