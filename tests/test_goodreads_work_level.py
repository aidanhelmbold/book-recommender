"""Goodreads nodes must be works, not editions.

Confirmed on the real dump: seeding "Dune" matched *five* separate nodes — "Dune",
"Dune (Dune Chronicles, #1)" three times over, and "Dune (Dune Chronicles #1)" —
each carrying a fifth of the book's similar-books evidence. The recommender walks
edges, so splitting one work across five nodes divides the signal it runs on, and
scores fall off a cliff after the two neighbours whichever node happened to be
picked as the seed.

The dump already solves this: every record carries both ``book_id`` (the edition)
and ``work_id`` (the work). Keying nodes on ``work_id`` collapses editions with no
fuzzy matching at all. ``similar_books`` cites *book_ids*, so each edition must
also register a ``goodreads_id -> work`` alias, which is what lets every
edition's edges land on the one node.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


from bookmap.sources.base import split_ref
from bookmap.sources.goodreads_ucsd import GoodreadsUCSDSource
from bookmap.store.db import Store


def _dump(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "books.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


def _record(book_id: str, work_id: str, title: str, similar: list[str] | None = None) -> dict:
    """One dump record, with the fields the real file actually carries."""
    return {
        "book_id": book_id,
        "work_id": work_id,
        "title": title,
        "title_without_series": title.split(" (")[0],
        "authors": [{"author_id": "58", "role": ""}],
        "isbn": "",
        "isbn13": "",
        "asin": "",
        "publication_year": "1965",
        "average_rating": "4.25",
        "ratings_count": "1000",
        "series": [],
        "similar_books": similar or [],
    }


THREE_DUNE_EDITIONS = [
    # Three editions of one work, exactly as the real dump presents them.
    _record("234225", "3634639", "Dune (Dune Chronicles, #1)", ["106"]),
    _record("44492285", "3634639", "Dune", ["106"]),
    _record("53732", "3634639", "Dune (Dune Chronicles #1)", []),
    # A different work that cites one of those editions by its book_id.
    _record("106", "3634640", "Dune Messiah", ["53732"]),
]


class TestWorkLevelIdentity:
    def test_editions_collapse_onto_one_node(self, tmp_path: Path) -> None:
        source = GoodreadsUCSDSource(_dump(tmp_path, THREE_DUNE_EDITIONS))
        work_ids = {book.work_id for book in source.iter_books()}
        # Three Dune editions plus Dune Messiah = two works, not four editions.
        assert len(work_ids) == 2, f"expected 2 works, got {sorted(work_ids)}"

    def test_falls_back_to_book_id_when_work_id_absent(self, tmp_path: Path) -> None:
        """Not every record carries a usable work_id; those must still ingest."""
        record = _record("999", "", "Orphan Edition")
        source = GoodreadsUCSDSource(_dump(tmp_path, [record]))
        books = list(source.iter_books())
        assert len(books) == 1
        assert "999" in books[0].work_id

    def test_every_edition_registers_its_book_id(self, tmp_path: Path) -> None:
        """similar_books cites book_ids, so each edition needs its own alias.

        Without this, an edge pointing at edition 53732 cannot find the work that
        edition belongs to, and it is dropped.
        """
        source = GoodreadsUCSDSource(_dump(tmp_path, THREE_DUNE_EDITIONS))
        with Store.open(tmp_path / "t.duckdb") as store:
            store.upsert_books(source.iter_books())
            store.insert_raw_edges(source.iter_edges())
            for book_id in ("234225", "44492285", "53732"):
                assert store.alias_lookup("goodreads_id", book_id) is not None, (
                    f"edition {book_id} has no alias, so edges citing it will drop"
                )

    def test_edges_from_any_edition_reach_the_same_work(self, tmp_path: Path) -> None:
        """The payoff: signal aimed at any edition lands on the one work node."""
        source = GoodreadsUCSDSource(_dump(tmp_path, THREE_DUNE_EDITIONS))
        with Store.open(tmp_path / "t.duckdb") as store:
            store.upsert_books(source.iter_books())
            store.insert_raw_edges(source.iter_edges())
            dropped = store.resolve_refs()
            assert dropped == 0, "every cited book_id is present, so nothing should drop"

            edges = list(store.iter_raw_edges())
            assert edges
            endpoints = {edge.src for edge in edges} | {edge.dst for edge in edges}
            # Two works, so at most two distinct endpoints -- not four.
            assert len(endpoints) == 2, f"edges span {len(endpoints)} nodes: {sorted(endpoints)}"

    def test_ingest_sql_agrees_with_the_streaming_path(self, tmp_path: Path) -> None:
        """The bulk path is what runs on the 9GB file; it must key ids identically."""
        dump = _dump(tmp_path, THREE_DUNE_EDITIONS)
        source = GoodreadsUCSDSource(dump)

        with Store.open(tmp_path / "sql.duckdb") as store:
            source.ingest_sql(store)
            sql_works = {b.work_id for b in store.iter_books()}
            sql_edges = {(e.src, e.dst, e.rank) for e in store.iter_raw_edges()}
            sql_titles = {b.title for b in store.iter_books()}

        with Store.open(tmp_path / "stream.duckdb") as store:
            store.upsert_books(source.iter_books())
            store.insert_raw_edges(source.iter_edges())
            stream_works = {b.work_id for b in store.iter_books()}
            stream_edges = {(e.src, e.dst, e.rank) for e in store.iter_raw_edges()}

        assert sql_works == stream_works
        assert sql_edges == stream_edges
        # Titles are deliberately *not* compared. Several editions now collapse
        # onto one row, so which one supplies the display title is a choice, and
        # the two paths make it differently: SQL picks the most-rated edition,
        # while the streaming upsert is last-write-wins. Both are defensible; the
        # invariant that matters is that they agree on the set of works and edges.
        known = {"Dune", "Dune Messiah", "Orphan Edition"}
        assert sql_titles <= known, sql_titles

    def test_refs_still_cite_book_ids(self, tmp_path: Path) -> None:
        """Edges are emitted against book_ids because that is what the dump says.

        Rewriting them to work_ids at emit time would need a lookup the streaming
        reader cannot do -- a record may cite an edition described later in the
        file.
        """
        source = GoodreadsUCSDSource(_dump(tmp_path, THREE_DUNE_EDITIONS))
        edges = list(source.iter_edges())
        assert edges
        assert all(split_ref(edge.dst)[0] == "goodreads_ucsd" for edge in edges)


class TestSeedResolutionAfterCollapse:
    def test_one_work_means_one_seed_match(self, tmp_path: Path) -> None:
        """The user-visible symptom: "Dune" should not offer five alternatives."""
        from bookmap.recommend import resolve_seeds

        source = GoodreadsUCSDSource(_dump(tmp_path, THREE_DUNE_EDITIONS))
        with Store.open(tmp_path / "t.duckdb") as store:
            store.upsert_books(source.iter_books())
            matches, unresolved = resolve_seeds(["Dune"], store)
            assert unresolved == []
            assert len(matches) == 1
            # One work in the table, so there is nothing to be ambiguous between.
            assert not matches[0].ambiguous, (
                f"still ambiguous between {matches[0].alternatives}"
            )
