"""UCSD Book Graph (Goodreads) ingest.

The richest legal source of Goodreads' own recommendations. Each record carries
a ``similar_books`` array -- literally the output of the "Readers also enjoyed"
recommender -- alongside metadata, for ~2.36M books.

Research-use licensed; see README. Goodreads shut down its public API in
December 2020, so this dump is the route to that data, not scraping.

Format: newline-delimited JSON, gzipped. DuckDB reads it natively, so the whole
adapter reduces to a query with ``UNNEST(similar_books)`` -- the list position
becomes the edge rank without a Python loop.
"""

from __future__ import annotations

from collections.abc import Iterator

from bookmap.models import Book, Edge, SourceName
from bookmap.sources.base import FileSource


class GoodreadsUCSDSource(FileSource):
    """Reads ``goodreads_books.json.gz`` from the UCSD Book Graph.

    Note the dump's shape: a book's ``authors`` field holds ``author_id``
    references, *not* names -- the names live in a separate
    ``goodreads_book_authors.json.gz``. Pass ``authors_path`` to resolve them;
    without it books are ingested with empty authors, which degrades both
    display and title-based identity matching.
    """

    name = SourceName.GOODREADS_UCSD

    def __init__(self, path, authors_path=None) -> None:  # noqa: ANN001
        super().__init__(path)
        self.authors_path = authors_path
        self._author_names: dict[str, str] | None = None

    def load_author_names(self) -> dict[str, str]:
        """Load the ``author_id -> name`` map, or an empty dict if unavailable.

        Cached: the author file is small enough to hold in memory (~800k rows)
        and is needed for every book record.
        """
        raise NotImplementedError

    def iter_books(self) -> Iterator[Book]:
        """Stream book records.

        Fields of interest: ``book_id``, ``title`` (prefer
        ``title_without_series`` when present), ``authors``, ``isbn13``,
        ``isbn``, ``asin``, ``publication_year``, ``average_rating``,
        ``ratings_count``, ``series``.

        Numeric fields arrive as *strings* in this dump and are frequently the
        empty string; they must be coerced defensively rather than cast. Records
        with an empty title are skipped -- they cannot be resolved or displayed.
        """
        raise NotImplementedError

    def iter_edges(self) -> Iterator[Edge]:
        """Stream ``similar_books`` as ranked ``GR_SIMILAR`` edges.

        Array position becomes 1-based ``rank``; endpoints are namespaced refs
        (see :func:`bookmap.sources.base.unresolved_ref`) because a record may
        reference a book that appears later in the file.
        """
        raise NotImplementedError

    def ingest_sql(self, store, *, max_rank: int = 50) -> tuple[int, int]:  # noqa: ANN001
        """Bulk-load via DuckDB, bypassing Python row iteration.

        Uses ``read_json(path, format='newline_delimited')`` plus
        ``UNNEST(similar_books) WITH ORDINALITY`` to populate ``books`` and
        ``edges_raw`` in two statements. This is the path used for the real 2GB
        dump; :meth:`iter_books` / :meth:`iter_edges` exist for small files,
        tests, and other backends.

        Returns ``(books_added, edges_added)``.
        """
        raise NotImplementedError
