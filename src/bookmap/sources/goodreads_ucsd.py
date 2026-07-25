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

import gzip
import io
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from bookmap.models import Book, Edge, EdgeKind, SourceName
from bookmap.sources.base import FileSource, unresolved_ref

WORK_ID_PREFIX = "gr-"
"""Namespace for work ids minted from a Goodreads book id.

Deliberately colon-free: edge endpoints carry ``source:raw_id`` refs, and a
work id containing a colon would be indistinguishable from an unresolved ref.
"""


def open_text(path: Path) -> io.TextIOBase:
    """Open a dump for reading, transparently handling gzip.

    The real dumps are gzipped and the fixtures are too; a plain-text copy is
    still common after a manual ``gunzip``, so both are accepted.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Stream newline-delimited JSON objects, skipping blank lines.

    Streams rather than materialising: the books dump is ~2GB gzipped.
    """
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield record


def text_or_none(value: object) -> str | None:
    """Normalise a dump value to a non-empty string, or ``None``.

    This dump uses the empty string, not null, for missing values -- an empty
    ISBN must read as "absent" rather than as an identifier that matches every
    other blank one.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def int_or_none(value: object) -> int | None:
    """Coerce a stringly-typed integer, mapping anything unparseable to ``None``.

    Never falls back to 0: a blank ``publication_year`` coerced to zero would
    sort thousands of books to the front of every chronological view and break
    year filtering, which is far worse than an honest null.
    """
    text = text_or_none(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        # Some rows carry "1965.0" rather than "1965".
        return int(float(text))
    except ValueError:
        return None


def float_or_none(value: object) -> float | None:
    """Coerce a stringly-typed float, mapping anything unparseable to ``None``."""
    text = text_or_none(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


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
        if self._author_names is None:
            names: dict[str, str] = {}
            if self.authors_path is not None:
                for record in iter_json_lines(Path(self.authors_path)):
                    author_id = text_or_none(record.get("author_id"))
                    author_name = text_or_none(record.get("name"))
                    if author_id and author_name:
                        names[author_id] = author_name
            self._author_names = names
        return self._author_names

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
        author_names = self.load_author_names()
        for record in iter_json_lines(self.path):
            book_id = text_or_none(record.get("book_id"))
            title = _record_title(record)
            if book_id is None or title is None:
                continue

            # The dump gives author ids only; unresolvable ids are dropped
            # rather than surfaced as raw numbers in the UI.
            authors: list[str] = []
            for entry in record.get("authors") or ():
                if not isinstance(entry, dict):
                    continue
                author_id = text_or_none(entry.get("author_id"))
                if author_id is not None and author_id in author_names:
                    authors.append(author_names[author_id])

            series = record.get("series") or ()
            yield Book(
                work_id=WORK_ID_PREFIX + book_id,
                title=title,
                authors=tuple(authors),
                year=int_or_none(record.get("publication_year")),
                # ISBN-10 (``isbn``) is passed over here: converting it is
                # identity's job, and Book has no field to carry it verbatim.
                isbn13=text_or_none(record.get("isbn13")),
                asin=text_or_none(record.get("asin")),
                goodreads_id=book_id,
                series=text_or_none(series[0]) if series else None,
                avg_rating=float_or_none(record.get("average_rating")),
                ratings_count=int_or_none(record.get("ratings_count")),
            )

    def iter_edges(self) -> Iterator[Edge]:
        """Stream ``similar_books`` as ranked ``GR_SIMILAR`` edges.

        Array position becomes 1-based ``rank``; endpoints are namespaced refs
        (see :func:`bookmap.sources.base.unresolved_ref`) because a record may
        reference a book that appears later in the file.
        """
        for record in iter_json_lines(self.path):
            book_id = text_or_none(record.get("book_id"))
            # A titleless record is not a node, so its recommendations are not
            # edges either -- keeping them would create phantom endpoints.
            if book_id is None or _record_title(record) is None:
                continue

            src = unresolved_ref(self.name, book_id)
            for position, raw_target in enumerate(record.get("similar_books") or (), start=1):
                target = text_or_none(raw_target)
                if target is None or target == book_id:
                    continue
                yield Edge(
                    src=src,
                    dst=unresolved_ref(self.name, target),
                    kind=EdgeKind.GR_SIMILAR,
                    rank=position,
                    source=self.name,
                )

    def ingest_sql(self, store, *, max_rank: int = 50) -> tuple[int, int]:  # noqa: ANN001
        """Bulk-load via DuckDB, bypassing Python row iteration.

        Uses ``read_json(path, format='newline_delimited')`` plus
        ``UNNEST(similar_books) WITH ORDINALITY`` to populate ``books`` and
        ``edges_raw`` in two statements. This is the path used for the real 2GB
        dump; :meth:`iter_books` / :meth:`iter_edges` exist for small files,
        tests, and other backends.

        Returns ``(books_added, edges_added)``.
        """
        conn = store.conn
        present = json_columns(conn, self.path)
        books_added = self._ingest_books_sql(conn, present)
        edges_added = self._ingest_edges_sql(conn, present, max_rank=max_rank)
        store.mark_ingest(
            self.name.value,
            str(self.path),
            books_added=books_added,
            edges_added=edges_added,
            completed=True,
        )
        return books_added, edges_added

    # -- SQL fast path -----------------------------------------------------

    def _ingest_books_sql(self, conn, present: Mapping[str, str]) -> int:  # noqa: ANN001
        """Insert ``books`` (and the ``aliases`` rows edge resolution needs).

        ``aliases`` is written here because :meth:`ingest_sql` bypasses
        ``Store.upsert_books``, which is what normally maintains it -- without
        it every namespaced edge ref would fail to resolve.

        ``books.title_norm`` is deliberately left NULL: it must match
        :func:`bookmap.identity.normalize_title` exactly or seed resolution
        starts disagreeing with itself, and re-deriving those rules in SQL is
        how that drift happens. The store owns that column.
        """
        params: list[object] = [str(self.path)]
        resolve_authors = self.authors_path is not None and "authors" in present
        if resolve_authors:
            # Resolve author ids in SQL as well, so the fast path and the
            # streaming path agree on the author tuple, order included.
            authors_cte = """
            author_ids AS (
                SELECT
                    kept.goodreads_id,
                    position,
                    NULLIF(TRIM(CAST(entry.author_id AS VARCHAR)), '') AS author_id
                FROM kept, UNNEST(kept.authors) WITH ORDINALITY AS unnested(entry, position)
            ),
            author_names AS (
                SELECT
                    CAST(author_id AS VARCHAR) AS author_id,
                    NULLIF(TRIM(CAST(name AS VARCHAR)), '') AS name
                FROM read_json(?, format='newline_delimited')
            ),
            resolved_authors AS (
                SELECT
                    author_ids.goodreads_id,
                    list(author_names.name ORDER BY author_ids.position) AS authors
                FROM author_ids
                JOIN author_names USING (author_id)
                WHERE author_names.name IS NOT NULL
                GROUP BY author_ids.goodreads_id
            ),
            """
            params.append(str(self.authors_path))
        else:
            # Without the companion file there are only ids to show, so books
            # are ingested authorless rather than displaying raw numbers.
            authors_cte = """
            resolved_authors AS (
                SELECT NULL AS goodreads_id, []::VARCHAR[] AS authors WHERE FALSE
            ),
            """

        # ``TRY_CAST`` throughout: every numeric field in this dump is a string
        # and frequently blank, and a hard CAST would abort the whole ingest on
        # the first empty year.
        series_col = sql_column(present, "series", fallback="[]::VARCHAR[]")
        sql = f"""
        WITH titled AS (
            SELECT
                NULLIF(TRIM(CAST({sql_column(present, "book_id")} AS VARCHAR)), '') AS goodreads_id,
                COALESCE(
                    NULLIF(TRIM(CAST({sql_column(present, "title_without_series")} AS VARCHAR)), ''),
                    NULLIF(TRIM(CAST({sql_column(present, "title")} AS VARCHAR)), '')
                ) AS title,
                {sql_column(present, "authors", fallback="[]")} AS authors,
                NULLIF(TRIM(CAST({sql_column(present, "isbn13")} AS VARCHAR)), '') AS isbn13,
                NULLIF(TRIM(CAST({sql_column(present, "asin")} AS VARCHAR)), '') AS asin,
                TRY_CAST(NULLIF(TRIM(CAST({sql_column(present, "publication_year")} AS VARCHAR)), '') AS INTEGER) AS year,
                TRY_CAST(NULLIF(TRIM(CAST({sql_column(present, "average_rating")} AS VARCHAR)), '') AS DOUBLE) AS avg_rating,
                TRY_CAST(NULLIF(TRIM(CAST({sql_column(present, "ratings_count")} AS VARCHAR)), '') AS BIGINT) AS ratings_count,
                CASE WHEN len({series_col}) > 0
                     THEN NULLIF(TRIM(CAST(({series_col})[1] AS VARCHAR)), '')
                END AS series
            FROM read_json(?, format='newline_delimited')
        ),
        kept AS (
            SELECT * FROM titled WHERE goodreads_id IS NOT NULL AND title IS NOT NULL
        ),
        {authors_cte}
        final AS (
            SELECT
                '{WORK_ID_PREFIX}' || kept.goodreads_id AS work_id,
                kept.title AS title,
                COALESCE(resolved_authors.authors, []::VARCHAR[]) AS authors,
                kept.year, kept.isbn13, kept.asin, kept.goodreads_id,
                kept.series, kept.avg_rating, kept.ratings_count
            FROM kept
            LEFT JOIN resolved_authors USING (goodreads_id)
        )
        """

        # DISTINCT ON: a dump may repeat a book id, and DuckDB refuses to update
        # the same conflicting row twice within one statement.
        books_sql = f"""
        {sql}
        INSERT INTO books
            (work_id, title, authors, year, isbn13, asin, goodreads_id, series,
             avg_rating, ratings_count)
        SELECT DISTINCT ON (work_id)
            work_id, title, authors, year, isbn13, asin, goodreads_id, series,
            avg_rating, ratings_count
        FROM final
        ON CONFLICT (work_id) DO UPDATE SET
            title = excluded.title,
            authors = excluded.authors,
            year = excluded.year,
            isbn13 = excluded.isbn13,
            asin = excluded.asin,
            goodreads_id = excluded.goodreads_id,
            series = excluded.series,
            avg_rating = excluded.avg_rating,
            ratings_count = excluded.ratings_count
        """
        books_added = _count_of(conn.execute(books_sql, params))

        aliases_sql = f"""
        {sql}
        INSERT INTO aliases (id_type, id_value, work_id)
        SELECT DISTINCT ON (id_type, id_value) id_type, id_value, work_id FROM (
            SELECT 'goodreads_id' AS id_type, goodreads_id AS id_value, work_id FROM final
            UNION ALL
            SELECT 'isbn13', isbn13, work_id FROM final WHERE isbn13 IS NOT NULL
            UNION ALL
            SELECT 'asin', asin, work_id FROM final WHERE asin IS NOT NULL
        )
        ON CONFLICT (id_type, id_value) DO UPDATE SET work_id = excluded.work_id
        """
        conn.execute(aliases_sql, params)
        return books_added

    def _ingest_edges_sql(self, conn, present: Mapping[str, str], *, max_rank: int) -> int:  # noqa: ANN001
        if "similar_books" not in present:
            return 0
        # Derived from unresolved_ref() rather than hardcoded, so the SQL and
        # streaming paths cannot drift apart on the ref format.
        ref_prefix = unresolved_ref(self.name, "\x00").removesuffix("\x00")
        edges_sql = f"""
        WITH kept AS (
            SELECT
                NULLIF(TRIM(CAST({sql_column(present, "book_id")} AS VARCHAR)), '') AS goodreads_id,
                similar_books
            FROM read_json(?, format='newline_delimited')
            WHERE NULLIF(TRIM(CAST({sql_column(present, "book_id")} AS VARCHAR)), '') IS NOT NULL
              AND COALESCE(
                    NULLIF(TRIM(CAST({sql_column(present, "title_without_series")} AS VARCHAR)), ''),
                    NULLIF(TRIM(CAST({sql_column(present, "title")} AS VARCHAR)), '')
                  ) IS NOT NULL
        ),
        exploded AS (
            SELECT
                kept.goodreads_id AS src_id,
                NULLIF(TRIM(CAST(target AS VARCHAR)), '') AS dst_id,
                position AS rank
            FROM kept, UNNEST(kept.similar_books) WITH ORDINALITY AS unnested(target, position)
            WHERE position <= ?
        )
        INSERT INTO edges_raw (src, dst, kind, rank, source)
        SELECT DISTINCT ON (src, dst, kind, source) src, dst, kind, rank, source FROM (
            SELECT
                '{ref_prefix}' || src_id AS src,
                '{ref_prefix}' || dst_id AS dst,
                '{EdgeKind.GR_SIMILAR.value}' AS kind,
                rank,
                '{self.name.value}' AS source
            FROM exploded
            WHERE dst_id IS NOT NULL AND dst_id <> src_id
        )
        ORDER BY rank
        ON CONFLICT (src, dst, kind, source) DO UPDATE SET rank = excluded.rank
        """
        return _count_of(conn.execute(edges_sql, [str(self.path), max_rank]))


def json_columns(conn, path: Path) -> dict[str, str]:  # noqa: ANN001
    """The ``column -> inferred type`` schema DuckDB reads a JSON dump as.

    A key absent from *every* record is not a column, and referencing it is a
    bind error rather than a NULL -- so the bulk queries are assembled against
    the schema that is actually there. This is what lets one query serve both a
    2018 and a 2023 export of the same dataset, whose keys differ.
    """
    described = conn.execute(
        "DESCRIBE SELECT * FROM read_json(?, format='newline_delimited')",
        [str(path)],
    ).fetchall()
    return {row[0]: row[1] for row in described}


def sql_column(present: Mapping[str, str], name: str, *, fallback: str = "NULL") -> str:
    """Reference ``name`` if the dump has it, else the fallback literal."""
    return name if name in present else fallback


def _record_title(record: dict[str, Any]) -> str | None:
    """The title to ingest, or ``None`` when the record has none.

    ``title_without_series`` wins when present so that "Dune (Dune Chronicles,
    #1)" becomes "Dune" -- series decoration otherwise defeats title matching
    against Amazon, which does not carry it.
    """
    return text_or_none(record.get("title_without_series")) or text_or_none(record.get("title"))


def _count_of(result) -> int:  # noqa: ANN001
    """Rows affected by a DuckDB DML statement."""
    row = result.fetchone()
    return int(row[0]) if row else 0
