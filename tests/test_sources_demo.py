"""Demo corpus loader.

The bundled corpus is hand-authored, and these tests assert that it says so.
Mislabelling it as scraped Amazon or Goodreads output would misrepresent where
the data came from, so provenance is checked, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bookmap.models import EdgeKind, SourceName
from bookmap.sources.demo import DEFAULT_DEMO_PATH, DemoSource


@pytest.fixture
def demo_file(tmp_path: Path) -> Path:
    payload = {
        "_provenance": "Hand-authored demo fixture. Not scraped from Amazon or Goodreads.",
        "books": [
            {
                "id": "dune",
                "title": "Dune",
                "authors": ["Frank Herbert"],
                "year": 1965,
                "subjects": ["space opera", "desert"],
            },
            {
                "id": "foundation",
                "title": "Foundation",
                "authors": ["Isaac Asimov"],
                "year": 1951,
                "subjects": ["space opera"],
            },
            {
                "id": "emma",
                "title": "Emma",
                "authors": ["Jane Austen"],
                "year": 1815,
                "subjects": ["regency"],
            },
        ],
        "edges": [
            {"src": "dune", "similar": ["foundation"], "kind": "gr_similar"},
            {"src": "foundation", "similar": ["dune", "emma"], "kind": "az_also_bought"},
        ],
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestDemoSource:
    def test_reads_books(self, demo_file: Path) -> None:
        books = list(DemoSource(demo_file).iter_books())
        assert {b.title for b in books} == {"Dune", "Foundation", "Emma"}

    def test_authors_and_years(self, demo_file: Path) -> None:
        books = {b.title: b for b in DemoSource(demo_file).iter_books()}
        assert books["Dune"].authors == ("Frank Herbert",)
        assert books["Dune"].year == 1965

    def test_subjects(self, demo_file: Path) -> None:
        books = {b.title: b for b in DemoSource(demo_file).iter_books()}
        assert "space opera" in books["Dune"].subjects

    def test_edge_count_and_kinds(self, demo_file: Path) -> None:
        edges = list(DemoSource(demo_file).iter_edges())
        assert len(edges) == 3
        assert {e.kind for e in edges} == {EdgeKind.GR_SIMILAR, EdgeKind.AZ_ALSO_BOUGHT}

    def test_list_position_becomes_rank(self, demo_file: Path) -> None:
        edges = [e for e in DemoSource(demo_file).iter_edges() if "foundation" in e.src]
        assert sorted(e.rank for e in edges) == [1, 2]

    def test_source_is_labelled_demo(self, demo_file: Path) -> None:
        """Provenance must survive into the store, so demo edges are never
        mistaken for real co-purchase data."""
        edges = list(DemoSource(demo_file).iter_edges())
        assert all(e.source == SourceName.DEMO for e in edges)

    def test_default_path_is_used_when_none_given(self) -> None:
        source = DemoSource()
        assert Path(source.path) == DEFAULT_DEMO_PATH

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DemoSource(tmp_path / "nope.json")


class TestBundledCorpus:
    """Assertions about the corpus that actually ships."""

    def test_exists(self) -> None:
        assert DEFAULT_DEMO_PATH.exists()

    def test_declares_its_provenance(self) -> None:
        """The file must state that it is hand-authored, not scraped."""
        payload = json.loads(DEFAULT_DEMO_PATH.read_text(encoding="utf-8"))
        provenance = payload.get("_provenance", "").lower()
        assert provenance
        assert "hand-authored" in provenance or "hand authored" in provenance

    def test_is_large_enough_to_be_interesting(self) -> None:
        """Needs enough books and clusters for community detection and MMR to
        have something to work with."""
        payload = json.loads(DEFAULT_DEMO_PATH.read_text(encoding="utf-8"))
        assert len(payload["books"]) >= 150

    def test_loads_without_error(self) -> None:
        books = list(DemoSource().iter_books())
        edges = list(DemoSource().iter_edges())
        assert len(books) >= 150
        assert len(edges) >= 500

    def test_every_edge_endpoint_is_a_known_book(self) -> None:
        """A dangling reference would become a titleless phantom node."""
        payload = json.loads(DEFAULT_DEMO_PATH.read_text(encoding="utf-8"))
        ids = {book["id"] for book in payload["books"]}
        for edge in payload["edges"]:
            assert edge["src"] in ids
            for target in edge["similar"]:
                assert target in ids, f"unknown edge target {target!r}"
