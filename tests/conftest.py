"""Shared fixtures.

The graphs here are small enough that the correct answer can be worked out by
hand, which is the point: they are oracles, not snapshots.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from bookmap.models import Book, Edge, EdgeKind, SourceName

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.duckdb"


@pytest.fixture
def store(db_path: Path):
    """An open, schema-applied store on a temp file."""
    from bookmap.store.db import Store

    with Store.open(db_path) as s:
        yield s


@pytest.fixture
def sample_books() -> list[Book]:
    """Four books, two authors, one shared author -- enough to exercise
    same-author filtering and multi-author identity."""
    return [
        Book(
            work_id="w-dune",
            title="Dune",
            authors=("Frank Herbert",),
            year=1965,
            isbn13="9780441013593",
            goodreads_id="234225",
        ),
        Book(
            work_id="w-dune-messiah",
            title="Dune Messiah",
            authors=("Frank Herbert",),
            year=1969,
            goodreads_id="106",
        ),
        Book(
            work_id="w-foundation",
            title="Foundation",
            authors=("Isaac Asimov",),
            year=1951,
            goodreads_id="29579",
        ),
        Book(
            work_id="w-hyperion",
            title="Hyperion",
            authors=("Dan Simmons",),
            year=1989,
            goodreads_id="77566",
        ),
    ]


@pytest.fixture
def sample_edges() -> list[Edge]:
    return [
        Edge("w-dune", "w-dune-messiah", EdgeKind.GR_SIMILAR, 1, SourceName.GOODREADS_UCSD),
        Edge("w-dune", "w-foundation", EdgeKind.GR_SIMILAR, 2, SourceName.GOODREADS_UCSD),
        Edge("w-foundation", "w-dune", EdgeKind.GR_SIMILAR, 1, SourceName.GOODREADS_UCSD),
        Edge("w-foundation", "w-hyperion", EdgeKind.AZ_ALSO_BOUGHT, 1, SourceName.AMAZON_META),
        Edge("w-hyperion", "w-dune", EdgeKind.AZ_ALSO_VIEWED, 3, SourceName.AMAZON_META),
    ]


@pytest.fixture
def populated_store(store, sample_books, sample_edges):
    """A store with books, raw edges, and a built fused graph."""
    from bookmap.graph.build import fuse_edges

    store.upsert_books(sample_books)
    store.insert_raw_edges(sample_edges)
    store.replace_fused_edges(fuse_edges(sample_edges))
    return store


@pytest.fixture
def two_cluster_pairs() -> list[tuple[str, str, float]]:
    """Two genre clusters joined by a single bridge book.

    Hand-built so that the expected behaviour is unambiguous: seeding inside one
    cluster must surface that cluster, and ``bridge`` must score highest on
    betweenness because every cross-cluster path goes through it.
    """
    left = ["sf1", "sf2", "sf3", "sf4"]
    right = ["lit1", "lit2", "lit3", "lit4"]
    pairs: list[tuple[str, str, float]] = []
    for group in (left, right):
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                pairs.append((a, b, 1.0))
    pairs.append(("sf1", "bridge", 1.0))
    pairs.append(("bridge", "lit1", 1.0))
    return pairs


@pytest.fixture
def goodreads_fixture(tmp_path: Path) -> Path:
    """A genuinely gzipped, newline-delimited Goodreads-format sample.

    Gzipped on purpose: the real dump is, DuckDB's ``read_json`` must handle it,
    and a plain-text fixture would let a broken gzip path pass.

    Mirrors the dump's real awkwardness -- numbers as strings, empty strings for
    missing values, authors as id references, and one titleless record that must
    be skipped.
    """
    records = [
        {
            "book_id": "234225",
            "title": "Dune (Dune Chronicles, #1)",
            "title_without_series": "Dune",
            "authors": [{"author_id": "58", "role": ""}],
            "isbn": "0441013597",
            "isbn13": "9780441013593",
            "asin": "",
            "publication_year": "1965",
            "average_rating": "4.25",
            "ratings_count": "801474",
            "series": ["45935"],
            "similar_books": ["106", "29579", "77566"],
        },
        {
            "book_id": "106",
            "title": "Dune Messiah",
            "title_without_series": "Dune Messiah",
            "authors": [{"author_id": "58", "role": ""}],
            "isbn": "",
            "isbn13": "",
            "asin": "B000FC1PWA",
            "publication_year": "",
            "average_rating": "3.87",
            "ratings_count": "",
            "series": [],
            "similar_books": ["234225"],
        },
        {
            "book_id": "29579",
            "title": "Foundation",
            "title_without_series": "Foundation",
            "authors": [{"author_id": "16667", "role": ""}],
            "isbn": "",
            "isbn13": "9780553293357",
            "asin": "",
            "publication_year": "1951",
            "average_rating": "4.22",
            "ratings_count": "500000",
            "series": [],
            "similar_books": ["234225", "77566"],
        },
        {
            "book_id": "999999",
            "title": "",
            "title_without_series": "",
            "authors": [],
            "isbn": "",
            "isbn13": "",
            "asin": "",
            "publication_year": "",
            "average_rating": "",
            "ratings_count": "",
            "series": [],
            "similar_books": ["234225"],
        },
    ]
    path = tmp_path / "goodreads_books.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def goodreads_authors_fixture(tmp_path: Path) -> Path:
    """The companion author-name file the books dump only references by id."""
    records = [
        {"author_id": "58", "name": "Frank Herbert", "average_rating": "4.01"},
        {"author_id": "16667", "name": "Isaac Asimov", "average_rating": "4.15"},
    ]
    path = tmp_path / "goodreads_book_authors.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def amazon_meta_fixture(tmp_path: Path) -> Path:
    """SNAP ``amazon-meta.txt`` sample: one book, one non-book, one discontinued.

    The non-book and discontinued records must both be skipped -- most of the
    real file is DVDs and music.
    """
    content = """Total items: 3

Id:   1
ASIN: 0827229534
  title: Patterns of Preaching: A Sermon Sampler
  group: Book
  salesrank: 396585
  similar: 5  0804215715  156101074X  0687023955  0687074231  082721619X
  categories: 2
   |Books[283155]|Subjects[1000]|Religion & Spirituality[22]
  reviews: total: 2  downloaded: 2  avg rating: 5
    2000-7-28  cutomer: A2JW67OY8U6HHK  rating: 5  votes:  10  helpful:   9

Id:   2
ASIN: 0738700797
  title: Candlemas: Feast of Flames
  group: DVD
  salesrank: 168596
  similar: 5  0738700827  1567184960  1567182836  0738700525  0738700940
  categories: 2
  reviews: total: 12  downloaded: 12  avg rating: 4.5

Id:   3
ASIN: 0486287785
  discontinued product

Id:   4
ASIN: 0804215715
  title: The Preaching Life
  group: Book
  salesrank: 500000
  similar: 2  0827229534  156101074X
  categories: 1
  reviews: total: 1  downloaded: 1  avg rating: 4
"""
    path = tmp_path / "amazon-meta.txt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def amazon_reviews_fixture(tmp_path: Path) -> Path:
    """McAuley-style book metadata with both also_buy and also_view."""
    records = [
        {
            "parent_asin": "0441013597",
            "title": "Dune",
            "author": {"name": "Frank Herbert"},
            "average_rating": 4.6,
            "rating_number": 12000,
            "categories": ["Books", "Science Fiction & Fantasy"],
            "also_buy": ["0441172717", "0441294677"],
            "also_view": ["0553293354"],
        },
        {
            "parent_asin": "0441172717",
            "title": "Dune Messiah",
            "author": {"name": "Frank Herbert"},
            "average_rating": 4.2,
            "rating_number": 3000,
            "categories": ["Books"],
            "also_buy": ["0441013597"],
            "also_view": [],
        },
    ]
    path = tmp_path / "meta_Books.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path
