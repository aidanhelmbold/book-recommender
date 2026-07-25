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

import asyncio
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence

import httpx

from bookmap.models import Book, Edge, EdgeKind, SourceName

API_BASE = "https://openlibrary.org"
DEFAULT_RATE_LIMIT = 10.0
"""Requests per second. Open Library asks for reasonable use; this is well
inside it."""

WORK_ID_PREFIX = "ol-"
"""Namespace for work ids minted from an Open Library key."""

_YEAR = re.compile(r"(1[0-9]{3}|20[0-9]{2})")


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
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        # One permit per request, released on a timer: enough to keep bursts
        # inside the advertised rate without serialising the whole batch.
        self._min_interval = 1.0 / rate_limit if rate_limit > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self.transport,
            headers={"User-Agent": "bookmap/0.1 (+https://github.com/bookmap)"},
        )

    async def _throttle(self) -> None:
        """Space requests out to at most ``rate_limit`` per second."""
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_slot = now + self._min_interval

    async def fetch_by_isbn(self, isbn13: str) -> Book | None:
        """Look up one book by ISBN-13. Returns None on 404."""
        async with self._client() as client:
            return await self._fetch_by_isbn(client, isbn13)

    async def _fetch_by_isbn(self, client: httpx.AsyncClient, isbn13: str) -> Book | None:
        """Fetch and parse one ISBN, retrying transient server errors.

        A 404 is a real answer -- the ISBN is simply not in Open Library -- so it
        is not retried. 5xx and transport errors are, because losing a record to
        a blip means it never gets enriched.
        """
        for attempt in range(1, max(self.max_retries, 1) + 1):
            await self._throttle()
            try:
                response = await client.get(f"/isbn/{isbn13}.json")
            except httpx.HTTPError:
                if attempt >= self.max_retries:
                    return None
                await self._backoff(attempt)
                continue

            if response.status_code == 404:
                return None
            if response.status_code >= 500:
                if attempt >= self.max_retries:
                    return None
                await self._backoff(attempt)
                continue
            if response.status_code >= 400:
                # 4xx other than 404 means the request itself is wrong; retrying
                # an identical request cannot fix it.
                return None

            try:
                payload = response.json()
            except ValueError:
                return None
            return parse_book(payload, isbn13=isbn13)
        return None

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff between retries, floored by the rate limit."""
        await asyncio.sleep(max(self._min_interval, 0.05) * (2 ** (attempt - 1)))

    async def fetch_many(self, isbn13s: Sequence[str]) -> list[Book]:
        """Look up many ISBNs concurrently, respecting the rate limit.

        Individual failures are skipped rather than aborting the batch -- a
        long enrichment run must not die on one bad identifier.
        """
        if not isbn13s:
            return []
        async with self._client() as client:
            results = await asyncio.gather(
                *(self._fetch_by_isbn(client, isbn) for isbn in isbn13s),
                return_exceptions=True,
            )
        # Exceptions are dropped alongside misses: one unparseable record must
        # not cost the other thousands in the batch.
        return [item for item in results if isinstance(item, Book)]

    async def enrich_store(self, store, *, limit: int | None = None) -> int:  # noqa: ANN001
        """Fill in missing subjects/years/covers for books already stored.

        Only touches rows with gaps, so re-running is cheap. Returns the number
        of books updated.
        """
        rows = store.conn.execute(
            """
            SELECT work_id, isbn13 FROM books
            WHERE isbn13 IS NOT NULL
              AND (year IS NULL OR subjects IS NULL OR len(subjects) = 0)
            ORDER BY ratings_count DESC NULLS LAST
            """
            + ("LIMIT ?" if limit is not None else ""),
            [limit] if limit is not None else [],
        ).fetchall()
        if not rows:
            return 0

        by_isbn = {isbn13: work_id for work_id, isbn13 in rows}
        fetched = await self.fetch_many(list(by_isbn))

        updated = 0
        for book in fetched:
            if book.isbn13 is None:
                continue
            work_id = by_isbn.get(book.isbn13)
            if work_id is None:
                continue
            # COALESCE: enrichment fills gaps, it does not overwrite what a
            # behavioural source already established.
            store.conn.execute(
                """
                UPDATE books SET
                    year = COALESCE(year, ?),
                    subjects = CASE
                        WHEN subjects IS NULL OR len(subjects) = 0 THEN ?
                        ELSE subjects
                    END,
                    openlibrary_id = COALESCE(openlibrary_id, ?)
                WHERE work_id = ?
                """,
                [book.year, list(book.subjects), book.openlibrary_id, work_id],
            )
            updated += 1
        return updated

    def iter_books(self) -> Iterator[Book]:
        """Not supported: Open Library is an enrichment source, not a corpus.

        Raises :class:`NotImplementedError` permanently -- it satisfies the
        Source protocol's shape but there is no bulk crawl to offer.

        Deliberately not a generator: the error must surface when the method is
        called, not lie dormant until something iterates the result.
        """
        raise NotImplementedError(
            "Open Library has no bulk corpus; use fetch_many() to enrich stored books"
        )

    def iter_edges(self) -> Iterator[Edge]:
        """Not supported; use :func:`subject_edges` over stored books instead."""
        raise NotImplementedError(
            "Open Library asserts no behavioural edges; use subject_edges() instead"
        )


def parse_book(payload: dict, *, isbn13: str | None = None) -> Book | None:
    """Build a :class:`Book` from an Open Library edition document.

    Returns ``None`` for a document with no usable title, since a titleless
    node cannot be displayed or matched.
    """
    if not isinstance(payload, dict):
        return None
    title = str(payload.get("title") or "").strip()
    if not title:
        return None

    key = str(payload.get("key") or "").strip()
    openlibrary_id = key.rsplit("/", 1)[-1] if key else None

    subjects = tuple(
        subject
        for raw in payload.get("subjects") or ()
        if (subject := str(raw).strip())
    )

    # The edition record carries author *keys*, not names; resolving them costs
    # one request each, so display names come from the dumps instead.
    work_id = WORK_ID_PREFIX + (openlibrary_id or isbn13 or title)
    return Book(
        work_id=work_id,
        title=title,
        year=_publish_year(payload.get("publish_date")),
        isbn13=isbn13 or _first_isbn13(payload),
        openlibrary_id=openlibrary_id,
        subjects=subjects,
    )


def _publish_year(value: object) -> int | None:
    """Pull a year out of Open Library's free-text ``publish_date``.

    The field is genuinely free text ("1965", "June 1, 1965", "n.d."), so a
    four-digit year is extracted rather than parsed.
    """
    if value is None:
        return None
    match = _YEAR.search(str(value))
    return int(match.group(1)) if match else None


def _first_isbn13(payload: dict) -> str | None:
    for raw in payload.get("isbn_13") or ():
        text = str(raw).strip()
        if text:
            return text
    return None


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
    # Inverted index rather than an all-pairs scan: only books that actually
    # share a subject are ever compared, which keeps this near-linear on a
    # corpus where most pairs share nothing.
    by_subject: dict[str, list[str]] = defaultdict(list)
    for book in books:
        for subject in dict.fromkeys(book.subjects):
            by_subject[subject].append(book.work_id)

    shared: dict[tuple[str, str], int] = defaultdict(int)
    for holders in by_subject.values():
        if len(holders) > max_subject_size or len(holders) < 2:
            continue
        for index, first in enumerate(holders):
            for second in holders[index + 1 :]:
                if first == second:
                    continue
                # Canonical ordering: the pair is undirected, so it must not be
                # counted twice under two spellings.
                shared[(first, second) if first < second else (second, first)] += 1

    for (src, dst), count in shared.items():
        if count >= min_shared:
            yield Edge(
                src=src,
                dst=dst,
                kind=EdgeKind.OL_SUBJECT,
                rank=1,
                source=SourceName.OPENLIBRARY,
            )
