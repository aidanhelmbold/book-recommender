"""Amazon co-purchase ingest -- the "customers also bought" signal.

Two upstream formats, one adapter each:

``SNAP amazon-meta.txt`` is a colon-delimited record format where each product
lists ``similar`` ASINs. Only ``group: Book`` records are kept.

``Amazon Reviews 2023`` (McAuley lab) is newline-delimited JSON metadata with
``also_buy`` and ``also_view`` arrays -- these map to different edge kinds,
because viewing together is much weaker evidence than buying together.

Both are research-licensed dumps. Amazon's Product Advertising API no longer
exposes similar-items, so this is the available route to co-purchase data.
"""

from __future__ import annotations

from collections.abc import Iterator

from bookmap.models import Book, Edge, SourceName
from bookmap.sources.base import FileSource


class AmazonMetaSource(FileSource):
    """Reads SNAP ``amazon-meta.txt`` (optionally gzipped).

    Record shape::

        Id:   1
        ASIN: 0827229534
        title: Patterns of Preaching: A Sermon Sampler
        group: Book
        salesrank: 396585
        similar: 5  0804215715  156101074X  0687023955
        categories: 2
        reviews: total: 2  downloaded: 2  avg rating: 5

    ``similar`` positions become edge ranks. Non-book groups (DVD, Music,
    Video) are skipped, which removes most of the file.
    """

    name = SourceName.AMAZON_META

    def iter_books(self) -> Iterator[Book]:
        raise NotImplementedError

    def iter_edges(self) -> Iterator[Edge]:
        """Yield ``AZ_ALSO_BOUGHT`` edges from ``similar`` lines."""
        raise NotImplementedError

    @staticmethod
    def parse_record(block: str) -> tuple[Book | None, list[str]]:
        """Parse one record block into a book and its similar ASINs.

        Returns ``(None, [])`` for discontinued products and non-book groups.
        Separated out so the fiddly text format is directly unit-testable.
        """
        raise NotImplementedError


class AmazonReviews2023Source(FileSource):
    """Reads McAuley Amazon Reviews 2023 ``meta_Books.jsonl(.gz)``.

    Fields of interest: ``parent_asin``, ``title``, ``author``, ``also_buy``,
    ``also_view``, ``average_rating``, ``rating_number``, ``categories``.
    """

    name = SourceName.AMAZON_REVIEWS_2023

    def iter_books(self) -> Iterator[Book]:
        raise NotImplementedError

    def iter_edges(self) -> Iterator[Edge]:
        """Yield ``AZ_ALSO_BOUGHT`` from ``also_buy`` and ``AZ_ALSO_VIEWED``
        from ``also_view``, each ranked by array position."""
        raise NotImplementedError

    def ingest_sql(self, store, *, max_rank: int = 50) -> tuple[int, int]:  # noqa: ANN001
        """Bulk-load via DuckDB ``read_json`` + ``UNNEST``, as with Goodreads."""
        raise NotImplementedError
