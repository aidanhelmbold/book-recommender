"""Open Library enrichment.

Open Library has no co-purchase or "also read" data, so it contributes no
behavioural edges. What it does provide, from a public keyless API, is clean
metadata: canonical work identifiers, subjects, cover images, publication years.
That fills gaps the dumps leave and gives the map something to display.

Shared-subject edges are available as a fallback for books with no behavioural
neighbours at all, but they are weighted an order of magnitude below real
co-purchase signal (see ``config.SOURCE_ALPHA``) because sharing the subject
"Fiction" means very little.

This is the one adapter that touches the network, so it is rate-limited and
retried politely.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from bookmap.models import Book, Edge, SourceName

API_BASE = "https://openlibrary.org"
DEFAULT_RATE_LIMIT = 10.0
"""Requests per second. Open Library asks for reasonable use; this is well
inside it."""


class OpenLibrarySource:
    """Async metadata enrichment against the Open Library API."""

    name = SourceName.OPENLIBRARY

    def __init__(
        self,
        *,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        timeout: float = 10.0,
        max_retries: int = 3,
        base_url: str = API_BASE,
        transport=None,  # noqa: ANN001 - httpx.AsyncBaseTransport
    ) -> None:
        """``transport`` overrides the HTTP transport, so tests exercise the real
        parsing and retry logic against a stub instead of the live API."""
        raise NotImplementedError

    async def fetch_by_isbn(self, isbn13: str) -> Book | None:
        """Look up one book by ISBN-13. Returns None on 404."""
        raise NotImplementedError

    async def fetch_many(self, isbn13s: Sequence[str]) -> list[Book]:
        """Look up many ISBNs concurrently, respecting the rate limit.

        Individual failures are skipped rather than aborting the batch -- a
        long enrichment run must not die on one bad identifier.
        """
        raise NotImplementedError

    async def enrich_store(self, store, *, limit: int | None = None) -> int:  # noqa: ANN001
        """Fill in missing subjects/years/covers for books already stored.

        Only touches rows with gaps, so re-running is cheap. Returns the number
        of books updated.
        """
        raise NotImplementedError

    def iter_books(self) -> Iterator[Book]:
        """Not supported: Open Library is an enrichment source, not a corpus.

        Raises :class:`NotImplementedError` permanently -- it satisfies the
        Source protocol's shape but there is no bulk crawl to offer.
        """
        raise NotImplementedError

    def iter_edges(self) -> Iterator[Edge]:
        """Not supported; use :func:`subject_edges` over stored books instead."""
        raise NotImplementedError


def subject_edges(
    books: Sequence[Book],
    *,
    min_shared: int = 2,
    max_subject_size: int = 200,
) -> Iterator[Edge]:
    """Emit ``OL_SUBJECT`` edges between books sharing enough subjects.

    Subjects held by more than ``max_subject_size`` books are ignored: broad
    tags like "Fiction" or "American literature" would otherwise generate
    enormous dense cliques that swamp the real signal.
    """
    raise NotImplementedError
