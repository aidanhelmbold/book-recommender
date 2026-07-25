"""Open Library adapter, exercised against a stub transport.

No test here touches the network: a stubbed ``httpx`` transport runs the real
parsing, retry, and concurrency logic while keeping the suite hermetic and fast.
"""

from __future__ import annotations

import httpx
import pytest

from bookmap.models import Book, EdgeKind
from bookmap.sources.openlibrary import OpenLibrarySource, subject_edges

DUNE_PAYLOAD = {
    "title": "Dune",
    "authors": [{"key": "/authors/OL79034A"}],
    "publish_date": "1965",
    "subjects": ["Science fiction", "Desert life", "Space colonies"],
    "key": "/books/OL1M",
}


def stub_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestFetchByISBN:
    @pytest.mark.asyncio
    async def test_parses_a_book(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=DUNE_PAYLOAD)

        source = OpenLibrarySource(transport=stub_transport(handler))
        book = await source.fetch_by_isbn("9780441013593")
        assert book is not None
        assert book.title == "Dune"
        assert "Science fiction" in book.subjects

    @pytest.mark.asyncio
    async def test_404_returns_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        source = OpenLibrarySource(transport=stub_transport(handler))
        assert await source.fetch_by_isbn("9780000000000") is None

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        """A transient 503 must not lose the record."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503)
            return httpx.Response(200, json=DUNE_PAYLOAD)

        source = OpenLibrarySource(transport=stub_transport(handler), max_retries=3)
        book = await source.fetch_by_isbn("9780441013593")
        assert book is not None
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        source = OpenLibrarySource(transport=stub_transport(handler), max_retries=2)
        assert await source.fetch_by_isbn("9780441013593") is None

    @pytest.mark.asyncio
    async def test_isbn_appears_in_the_request(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json=DUNE_PAYLOAD)

        source = OpenLibrarySource(transport=stub_transport(handler))
        await source.fetch_by_isbn("9780441013593")
        assert any("9780441013593" in url for url in seen)


class TestFetchMany:
    @pytest.mark.asyncio
    async def test_fetches_all(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=DUNE_PAYLOAD)

        source = OpenLibrarySource(transport=stub_transport(handler))
        books = await source.fetch_many(["9780441013593", "9780553293357"])
        assert len(books) == 2

    @pytest.mark.asyncio
    async def test_one_failure_does_not_abort_the_batch(self) -> None:
        """A long enrichment run must survive a single bad identifier."""
        def handler(request: httpx.Request) -> httpx.Response:
            if "0000" in str(request.url):
                return httpx.Response(500)
            return httpx.Response(200, json=DUNE_PAYLOAD)

        source = OpenLibrarySource(transport=stub_transport(handler), max_retries=1)
        books = await source.fetch_many(["9780441013593", "9780000000000"])
        assert len(books) == 1

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("should not be called")

        source = OpenLibrarySource(transport=stub_transport(handler))
        assert await source.fetch_many([]) == []


class TestUnsupportedBulkInterface:
    def test_iter_books_is_not_supported(self) -> None:
        """Open Library is an enrichment source; there is no bulk corpus to walk."""
        source = OpenLibrarySource(transport=stub_transport(lambda r: httpx.Response(200)))
        with pytest.raises(NotImplementedError):
            list(source.iter_books())

    def test_iter_edges_is_not_supported(self) -> None:
        source = OpenLibrarySource(transport=stub_transport(lambda r: httpx.Response(200)))
        with pytest.raises(NotImplementedError):
            list(source.iter_edges())


class TestSubjectEdges:
    def test_links_books_sharing_enough_subjects(self) -> None:
        books = [
            Book(work_id="w-a", title="A", subjects=("space opera", "hard sf", "aliens")),
            Book(work_id="w-b", title="B", subjects=("space opera", "hard sf")),
        ]
        edges = list(subject_edges(books, min_shared=2))
        assert len(edges) == 1
        assert edges[0].kind == EdgeKind.OL_SUBJECT

    def test_respects_min_shared(self) -> None:
        books = [
            Book(work_id="w-a", title="A", subjects=("space opera", "aliens")),
            Book(work_id="w-b", title="B", subjects=("space opera", "romance")),
        ]
        assert list(subject_edges(books, min_shared=2)) == []
        assert len(list(subject_edges(books, min_shared=1))) == 1

    def test_broad_subjects_are_ignored(self) -> None:
        """"Fiction" on 300 books would generate a 300-clique.

        Those cliques swamp real co-purchase signal and make every literary novel
        equidistant from every other, so over-broad subjects are dropped.
        """
        books = [
            Book(work_id=f"w-{i}", title=f"T{i}", subjects=("fiction", "general"))
            for i in range(10)
        ]
        assert list(subject_edges(books, min_shared=2, max_subject_size=5)) == []

    def test_no_self_loops(self) -> None:
        books = [Book(work_id="w-a", title="A", subjects=("x", "y"))]
        assert list(subject_edges(books, min_shared=1)) == []

    def test_empty_input(self) -> None:
        assert list(subject_edges([])) == []
