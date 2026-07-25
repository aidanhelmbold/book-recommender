"""Goodreads UCSD dump adapter.

The fixture reproduces the dump's real awkwardness on purpose: numbers arrive as
strings, missing values are empty strings rather than null, author names live in
a separate file, and a few records have no title at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bookmap.models import EdgeKind
from bookmap.sources.base import split_ref
from bookmap.sources.goodreads_ucsd import GoodreadsUCSDSource


class TestIterBooks:
    def test_reads_gzipped_ndjson(self, goodreads_fixture: Path) -> None:
        books = list(GoodreadsUCSDSource(goodreads_fixture).iter_books())
        assert len(books) == 3  # the titleless record is skipped

    def test_skips_titleless_records(self, goodreads_fixture: Path) -> None:
        books = list(GoodreadsUCSDSource(goodreads_fixture).iter_books())
        assert all(book.title for book in books)
        assert "999999" not in {book.goodreads_id for book in books}

    def test_prefers_title_without_series(self, goodreads_fixture: Path) -> None:
        """"Dune (Dune Chronicles, #1)" should ingest as "Dune"."""
        books = {b.goodreads_id: b for b in GoodreadsUCSDSource(goodreads_fixture).iter_books()}
        assert books["234225"].title == "Dune"

    def test_coerces_stringly_typed_numbers(self, goodreads_fixture: Path) -> None:
        books = {b.goodreads_id: b for b in GoodreadsUCSDSource(goodreads_fixture).iter_books()}
        dune = books["234225"]
        assert dune.year == 1965
        assert dune.avg_rating == pytest.approx(4.25)
        assert dune.ratings_count == 801474

    def test_empty_strings_become_none_not_zero(self, goodreads_fixture: Path) -> None:
        """A missing year must not silently become year 0.

        Coercing blanks to zero would put thousands of books at the start of any
        chronological view and corrupt any year-based filtering.
        """
        books = {b.goodreads_id: b for b in GoodreadsUCSDSource(goodreads_fixture).iter_books()}
        messiah = books["106"]
        assert messiah.year is None
        assert messiah.ratings_count is None
        assert messiah.isbn13 is None

    def test_captures_identifiers(self, goodreads_fixture: Path) -> None:
        books = {b.goodreads_id: b for b in GoodreadsUCSDSource(goodreads_fixture).iter_books()}
        assert books["234225"].isbn13 == "9780441013593"
        assert books["106"].asin == "B000FC1PWA"

    def test_authors_empty_without_the_companion_file(self, goodreads_fixture: Path) -> None:
        """The books dump only carries author *ids*."""
        books = list(GoodreadsUCSDSource(goodreads_fixture).iter_books())
        assert all(book.authors == () for book in books)

    def test_authors_resolved_with_the_companion_file(
        self, goodreads_fixture: Path, goodreads_authors_fixture: Path
    ) -> None:
        source = GoodreadsUCSDSource(goodreads_fixture, goodreads_authors_fixture)
        books = {b.goodreads_id: b for b in source.iter_books()}
        assert books["234225"].authors == ("Frank Herbert",)
        assert books["29579"].authors == ("Isaac Asimov",)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            GoodreadsUCSDSource(tmp_path / "nope.json.gz")


class TestLoadAuthorNames:
    def test_maps_ids_to_names(self, goodreads_fixture: Path, goodreads_authors_fixture: Path) -> None:
        source = GoodreadsUCSDSource(goodreads_fixture, goodreads_authors_fixture)
        assert source.load_author_names()["58"] == "Frank Herbert"

    def test_empty_without_a_path(self, goodreads_fixture: Path) -> None:
        assert GoodreadsUCSDSource(goodreads_fixture).load_author_names() == {}


class TestIterEdges:
    def test_edge_count(self, goodreads_fixture: Path) -> None:
        """3 + 1 + 2 from the three titled records; the titleless one contributes none."""
        edges = list(GoodreadsUCSDSource(goodreads_fixture).iter_edges())
        assert len(edges) == 6

    def test_kind_is_gr_similar(self, goodreads_fixture: Path) -> None:
        edges = list(GoodreadsUCSDSource(goodreads_fixture).iter_edges())
        assert all(edge.kind == EdgeKind.GR_SIMILAR for edge in edges)

    def test_array_position_becomes_rank(self, goodreads_fixture: Path) -> None:
        """similar_books for 234225 is [106, 29579, 77566] -> ranks 1, 2, 3."""
        edges = [
            e
            for e in GoodreadsUCSDSource(goodreads_fixture).iter_edges()
            if split_ref(e.src)[1] == "234225"
        ]
        assert {split_ref(e.dst)[1]: e.rank for e in edges} == {
            "106": 1,
            "29579": 2,
            "77566": 3,
        }

    def test_endpoints_are_namespaced_refs(self, goodreads_fixture: Path) -> None:
        """A record may cite a book that appears later in the file, so ids cannot
        be resolved during streaming."""
        edge = next(iter(GoodreadsUCSDSource(goodreads_fixture).iter_edges()))
        source_name, raw_id = split_ref(edge.src)
        assert source_name == "goodreads_ucsd"
        assert raw_id.isdigit()

    def test_titleless_record_contributes_no_edges(self, goodreads_fixture: Path) -> None:
        edges = list(GoodreadsUCSDSource(goodreads_fixture).iter_edges())
        assert all(split_ref(e.src)[1] != "999999" for e in edges)


class TestIngestSQL:
    def test_populates_the_store(self, goodreads_fixture: Path, goodreads_authors_fixture: Path, store) -> None:
        source = GoodreadsUCSDSource(goodreads_fixture, goodreads_authors_fixture)
        books_added, edges_added = source.ingest_sql(store)
        assert books_added == 3
        assert edges_added == 6

    def test_agrees_with_the_streaming_path(
        self, goodreads_fixture: Path, goodreads_authors_fixture: Path, store, db_path
    ) -> None:
        """The SQL fast path and the Python path must produce the same graph.

        The SQL path is what runs on the real 2GB dump, so a divergence between
        them would only ever show up in production.
        """
        from bookmap.store.db import Store

        source = GoodreadsUCSDSource(goodreads_fixture, goodreads_authors_fixture)
        source.ingest_sql(store)
        sql_books = {(b.work_id, b.title, b.authors) for b in store.iter_books()}
        sql_edges = {(e.src, e.dst, e.kind, e.rank) for e in store.iter_raw_edges()}

        with Store.open(db_path.parent / "streamed.duckdb") as other:
            other.upsert_books(source.iter_books())
            other.insert_raw_edges(source.iter_edges())
            stream_books = {(b.work_id, b.title, b.authors) for b in other.iter_books()}
            stream_edges = {(e.src, e.dst, e.kind, e.rank) for e in other.iter_raw_edges()}

        assert sql_books == stream_books
        assert sql_edges == stream_edges

    def test_unresolvable_references_drop_on_resolve(
        self, goodreads_fixture: Path, goodreads_authors_fixture: Path, store
    ) -> None:
        """Hyperion (77566) is cited but never described, so its edges are not real.

        The real dump cites books outside its own corpus constantly; keeping those
        would fill the map with titleless phantom nodes.
        """
        source = GoodreadsUCSDSource(goodreads_fixture, goodreads_authors_fixture)
        source.ingest_sql(store)
        dropped = store.resolve_refs()
        assert dropped == 2
        assert store.counts()["edges_raw"] == 4

    def test_is_idempotent(self, goodreads_fixture: Path, goodreads_authors_fixture: Path, store) -> None:
        source = GoodreadsUCSDSource(goodreads_fixture, goodreads_authors_fixture)
        source.ingest_sql(store)
        source.ingest_sql(store)
        assert store.counts()["books"] == 3
        assert store.counts()["edges_raw"] == 6

    def test_max_rank_truncates(self, goodreads_fixture: Path, store) -> None:
        source = GoodreadsUCSDSource(goodreads_fixture)
        _, edges_added = source.ingest_sql(store, max_rank=1)
        # One edge per record with a non-empty similar_books list.
        assert edges_added == 3
