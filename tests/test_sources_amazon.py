"""Amazon co-purchase adapters.

``amazon-meta.txt`` is a hand-rolled text format, so ``parse_record`` is tested
directly -- most of the real file is DVDs and music that must be filtered out
before anything else happens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bookmap.models import EdgeKind
from bookmap.sources.amazon_meta import AmazonMetaSource, AmazonReviews2023Source
from bookmap.sources.base import split_ref


class TestParseRecord:
    def test_parses_a_book(self) -> None:
        block = """Id:   1
ASIN: 0827229534
  title: Patterns of Preaching: A Sermon Sampler
  group: Book
  salesrank: 396585
  similar: 5  0804215715  156101074X  0687023955  0687074231  082721619X
  categories: 2
  reviews: total: 2  downloaded: 2  avg rating: 5
"""
        book, similar = AmazonMetaSource.parse_record(block)
        assert book is not None
        assert book.title == "Patterns of Preaching: A Sermon Sampler"
        assert book.asin == "0827229534"
        assert similar == [
            "0804215715",
            "156101074X",
            "0687023955",
            "0687074231",
            "082721619X",
        ]

    def test_skips_non_book_groups(self) -> None:
        block = """Id:   2
ASIN: 0738700797
  title: Candlemas: Feast of Flames
  group: DVD
  similar: 2  0738700827  1567184960
"""
        book, similar = AmazonMetaSource.parse_record(block)
        assert book is None
        assert similar == []

    def test_skips_discontinued_products(self) -> None:
        book, similar = AmazonMetaSource.parse_record("Id:   3\nASIN: 0486287785\n  discontinued product\n")
        assert book is None
        assert similar == []

    def test_handles_a_book_with_no_similar_line(self) -> None:
        block = "Id: 9\nASIN: 0827229534\n  title: Lonely Book\n  group: Book\n"
        book, similar = AmazonMetaSource.parse_record(block)
        assert book is not None
        assert similar == []

    def test_title_containing_a_colon_survives(self) -> None:
        """Titles routinely contain colons; naive splitting truncates them."""
        block = "Id: 1\nASIN: 0000000000\n  title: Dune: The Graphic Novel\n  group: Book\n"
        book, _ = AmazonMetaSource.parse_record(block)
        assert book is not None
        assert book.title == "Dune: The Graphic Novel"

    def test_empty_block(self) -> None:
        assert AmazonMetaSource.parse_record("") == (None, [])


class TestAmazonMetaSource:
    def test_only_books_are_ingested(self, amazon_meta_fixture: Path) -> None:
        books = list(AmazonMetaSource(amazon_meta_fixture).iter_books())
        assert {b.asin for b in books} == {"0827229534", "0804215715"}

    def test_edge_count(self, amazon_meta_fixture: Path) -> None:
        """5 similar ASINs on the first book plus 2 on the second."""
        edges = list(AmazonMetaSource(amazon_meta_fixture).iter_edges())
        assert len(edges) == 7

    def test_edges_are_also_bought(self, amazon_meta_fixture: Path) -> None:
        edges = list(AmazonMetaSource(amazon_meta_fixture).iter_edges())
        assert all(e.kind == EdgeKind.AZ_ALSO_BOUGHT for e in edges)

    def test_list_position_becomes_rank(self, amazon_meta_fixture: Path) -> None:
        edges = [
            e
            for e in AmazonMetaSource(amazon_meta_fixture).iter_edges()
            if split_ref(e.src)[1] == "0827229534"
        ]
        by_rank = {e.rank: split_ref(e.dst)[1] for e in edges}
        assert by_rank[1] == "0804215715"
        assert by_rank[2] == "156101074X"

    def test_no_edges_from_the_dvd_record(self, amazon_meta_fixture: Path) -> None:
        edges = list(AmazonMetaSource(amazon_meta_fixture).iter_edges())
        assert all(split_ref(e.src)[1] != "0738700797" for e in edges)

    def test_refs_are_namespaced_to_asin(self, amazon_meta_fixture: Path) -> None:
        edge = next(iter(AmazonMetaSource(amazon_meta_fixture).iter_edges()))
        assert split_ref(edge.src)[0] == "amazon_meta"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AmazonMetaSource(tmp_path / "nope.txt")


class TestAmazonReviews2023Source:
    def test_reads_books(self, amazon_reviews_fixture: Path) -> None:
        books = list(AmazonReviews2023Source(amazon_reviews_fixture).iter_books())
        assert {b.title for b in books} == {"Dune", "Dune Messiah"}

    def test_author_is_extracted_from_the_nested_field(self, amazon_reviews_fixture: Path) -> None:
        books = {b.title: b for b in AmazonReviews2023Source(amazon_reviews_fixture).iter_books()}
        assert books["Dune"].authors == ("Frank Herbert",)

    def test_ratings_are_captured(self, amazon_reviews_fixture: Path) -> None:
        books = {b.title: b for b in AmazonReviews2023Source(amazon_reviews_fixture).iter_books()}
        assert books["Dune"].avg_rating == pytest.approx(4.6)
        assert books["Dune"].ratings_count == 12000

    def test_also_buy_and_also_view_are_different_kinds(self, amazon_reviews_fixture: Path) -> None:
        """Viewing together is far weaker evidence than buying together.

        Collapsing them would let browsing noise carry co-purchase weight.
        """
        edges = list(AmazonReviews2023Source(amazon_reviews_fixture).iter_edges())
        kinds = {e.kind for e in edges}
        assert kinds == {EdgeKind.AZ_ALSO_BOUGHT, EdgeKind.AZ_ALSO_VIEWED}

    def test_edge_counts_per_kind(self, amazon_reviews_fixture: Path) -> None:
        edges = list(AmazonReviews2023Source(amazon_reviews_fixture).iter_edges())
        bought = [e for e in edges if e.kind == EdgeKind.AZ_ALSO_BOUGHT]
        viewed = [e for e in edges if e.kind == EdgeKind.AZ_ALSO_VIEWED]
        assert len(bought) == 3
        assert len(viewed) == 1

    def test_ranks_start_at_one_per_list(self, amazon_reviews_fixture: Path) -> None:
        edges = [
            e
            for e in AmazonReviews2023Source(amazon_reviews_fixture).iter_edges()
            if split_ref(e.src)[1] == "0441013597" and e.kind == EdgeKind.AZ_ALSO_BOUGHT
        ]
        assert sorted(e.rank for e in edges) == [1, 2]

    def test_tolerates_bought_together_key(self, tmp_path: Path) -> None:
        """The 2023 release uses ``bought_together`` where 2018 used ``also_buy``.

        The adapter accepts either so a version change does not silently yield an
        edgeless graph.
        """
        import json

        path = tmp_path / "meta.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "parent_asin": "A1",
                        "title": "One",
                        "author": {"name": "X"},
                        "bought_together": ["A2"],
                    }
                )
                + "\n"
            )
        edges = list(AmazonReviews2023Source(path).iter_edges())
        assert len(edges) == 1
        assert edges[0].kind == EdgeKind.AZ_ALSO_BOUGHT

    def test_missing_edge_keys_yield_no_edges(self, tmp_path: Path) -> None:
        import json

        path = tmp_path / "meta.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"parent_asin": "A1", "title": "One"}) + "\n")
        assert list(AmazonReviews2023Source(path).iter_edges()) == []

    def test_ingest_sql_matches_streaming(self, amazon_reviews_fixture: Path, store, db_path) -> None:
        from bookmap.store.db import Store

        source = AmazonReviews2023Source(amazon_reviews_fixture)
        source.ingest_sql(store)
        sql_edges = {(e.src, e.dst, e.kind, e.rank) for e in store.iter_raw_edges()}
        with Store.open(db_path.parent / "streamed.duckdb") as other:
            other.upsert_books(source.iter_books())
            other.insert_raw_edges(source.iter_edges())
            stream_edges = {(e.src, e.dst, e.kind, e.rank) for e in other.iter_raw_edges()}
        assert sql_edges == stream_edges
