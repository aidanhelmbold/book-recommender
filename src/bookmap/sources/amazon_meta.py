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

from collections.abc import Iterator, Mapping

from bookmap.models import Book, Edge, EdgeKind, SourceName
from bookmap.sources.base import FileSource, unresolved_ref
from bookmap.sources.goodreads_ucsd import (
    float_or_none,
    int_or_none,
    iter_json_lines,
    json_columns,
    open_text,
    sql_column,
    text_or_none,
)
# The dump-reading and coercion helpers live in the Goodreads adapter because it
# was the first newline-delimited-JSON source; they are format utilities, not
# Goodreads-specific, and are shared rather than duplicated here.

WORK_ID_PREFIX = "az-"
"""Namespace for work ids minted from an ASIN.

Shared by both Amazon adapters on purpose: the 2003 SNAP dump and the 2023
release key on the same ASINs, so the same product must land on the same node
rather than being ingested twice.
"""

BOUGHT_KEYS = ("also_buy", "bought_together")
"""``also_buy`` (2018) and ``bought_together`` (2023) mean the same thing.

Accepting both matters: reading only the 2018 key against a 2023 file yields an
edgeless graph, which looks like a working ingest rather than a failed one.
"""

VIEWED_KEYS = ("also_view",)


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
        for block in self._iter_blocks():
            book, _ = self.parse_record(block)
            if book is not None:
                yield book

    def iter_edges(self) -> Iterator[Edge]:
        """Yield ``AZ_ALSO_BOUGHT`` edges from ``similar`` lines."""
        for block in self._iter_blocks():
            book, similar = self.parse_record(block)
            # ``parse_record`` returns no ASINs for skipped records, so DVDs and
            # discontinued products contribute no edges either.
            if book is None or book.asin is None:
                continue
            src = unresolved_ref(self.name, book.asin)
            for position, target in enumerate(similar, start=1):
                if target == book.asin:
                    continue
                yield Edge(
                    src=src,
                    dst=unresolved_ref(self.name, target),
                    kind=EdgeKind.AZ_ALSO_BOUGHT,
                    rank=position,
                    source=self.name,
                )

    def _iter_blocks(self) -> Iterator[str]:
        """Split the file into record blocks on blank lines.

        Streamed rather than read whole: the real file is ~1GB of text, and the
        vast majority of the blocks are thrown away.
        """
        with open_text(self.path) as handle:
            lines: list[str] = []
            for line in handle:
                if line.strip():
                    lines.append(line)
                    continue
                if lines:
                    yield "".join(lines)
                    lines = []
            if lines:
                yield "".join(lines)

    @staticmethod
    def parse_record(block: str) -> tuple[Book | None, list[str]]:
        """Parse one record block into a book and its similar ASINs.

        Returns ``(None, [])`` for discontinued products and non-book groups.
        Separated out so the fiddly text format is directly unit-testable.
        """
        asin: str | None = None
        title: str | None = None
        group: str | None = None
        similar: list[str] = []
        salesrank: int | None = None
        avg_rating: float | None = None
        ratings_count: int | None = None

        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line == "discontinued product":
                # A delisted product has no metadata at all, and its ASIN is
                # not a node anyone can be recommended.
                return None, []

            key, separator, value = line.partition(":")
            if not separator:
                continue
            key = key.strip().lower()
            # Only the first colon separates key from value -- titles such as
            # "Dune: The Graphic Novel" contain their own.
            value = value.strip()

            if key == "asin":
                asin = text_or_none(value)
            elif key == "title":
                title = text_or_none(value)
            elif key == "group":
                group = value.strip().lower()
            elif key == "salesrank":
                salesrank = int_or_none(value)
            elif key == "similar":
                # "similar: 5  ASIN1  ASIN2 ..." -- the leading count is
                # redundant and occasionally disagrees with the list, so the
                # list itself is trusted.
                similar = value.split()[1:] if value.split() else []
            elif key == "reviews":
                avg_rating = _reviews_field(value, "avg rating", float_or_none)
                ratings_count = _reviews_field(value, "total", int_or_none)

        if asin is None or title is None or group != "book":
            return None, []

        return (
            Book(
                work_id=WORK_ID_PREFIX + asin,
                title=title,
                asin=asin,
                avg_rating=avg_rating,
                ratings_count=ratings_count,
                # salesrank is read but has nowhere to live on Book; it is kept
                # out of the graph rather than smuggled into ratings_count.
            ),
            similar,
        )


class AmazonReviews2023Source(FileSource):
    """Reads McAuley Amazon Reviews 2023 ``meta_Books.jsonl(.gz)``.

    Fields of interest: ``parent_asin``, ``title``, ``author``, ``also_buy``,
    ``also_view``, ``average_rating``, ``rating_number``, ``categories``.
    """

    name = SourceName.AMAZON_REVIEWS_2023

    def iter_books(self) -> Iterator[Book]:
        for record in iter_json_lines(self.path):
            asin = text_or_none(record.get("parent_asin")) or text_or_none(record.get("asin"))
            title = text_or_none(record.get("title"))
            if asin is None or title is None:
                continue
            yield Book(
                work_id=WORK_ID_PREFIX + asin,
                title=title,
                authors=_author_names(record.get("author")),
                asin=asin,
                avg_rating=float_or_none(record.get("average_rating")),
                ratings_count=int_or_none(record.get("rating_number")),
                subjects=tuple(
                    subject
                    for raw in record.get("categories") or ()
                    if (subject := text_or_none(raw)) is not None
                ),
            )

    def iter_edges(self) -> Iterator[Edge]:
        """Yield ``AZ_ALSO_BOUGHT`` from ``also_buy`` and ``AZ_ALSO_VIEWED``
        from ``also_view``, each ranked by array position."""
        for record in iter_json_lines(self.path):
            asin = text_or_none(record.get("parent_asin")) or text_or_none(record.get("asin"))
            if asin is None or text_or_none(record.get("title")) is None:
                continue
            src = unresolved_ref(self.name, asin)
            for keys, kind in ((BOUGHT_KEYS, EdgeKind.AZ_ALSO_BOUGHT), (VIEWED_KEYS, EdgeKind.AZ_ALSO_VIEWED)):
                targets = _first_list(record, keys)
                # Rank restarts per list: rank 1 of "also viewed" is not a
                # continuation of "also bought", it is a separate ordering.
                for position, raw_target in enumerate(targets, start=1):
                    target = text_or_none(raw_target)
                    if target is None or target == asin:
                        continue
                    yield Edge(
                        src=src,
                        dst=unresolved_ref(self.name, target),
                        kind=kind,
                        rank=position,
                        source=self.name,
                    )

    def ingest_sql(self, store, *, max_rank: int = 50) -> tuple[int, int]:  # noqa: ANN001
        """Bulk-load via DuckDB ``read_json`` + ``UNNEST``, as with Goodreads."""
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

    def _books_cte(self, present: Mapping[str, str]) -> str:
        asin_col = sql_column(present, "parent_asin", fallback=sql_column(present, "asin"))
        categories = sql_column(present, "categories", fallback="[]::VARCHAR[]")
        # Only the nested ``{"author": {"name": ...}}`` shape is unpacked in SQL:
        # DuckDB infers one concrete type per column, so a file whose author is a
        # bare string simply has no ``.name`` to read. iter_books() is the path
        # that tolerates every historical variant.
        if present.get("author", "").startswith("STRUCT") and "name" in present["author"]:
            author_expr = """
                    CASE
                        WHEN NULLIF(TRIM(CAST(author.name AS VARCHAR)), '') IS NULL
                        THEN []::VARCHAR[]
                        ELSE [CAST(author.name AS VARCHAR)]
                    END
            """
        elif present.get("author") == "VARCHAR":
            author_expr = """
                    CASE
                        WHEN NULLIF(TRIM(CAST(author AS VARCHAR)), '') IS NULL
                        THEN []::VARCHAR[]
                        ELSE [CAST(author AS VARCHAR)]
                    END
            """
        else:
            author_expr = "[]::VARCHAR[]"
        return f"""
        WITH final AS (
            SELECT
                '{WORK_ID_PREFIX}' || asin AS work_id,
                title, authors, asin, avg_rating, ratings_count, subjects
            FROM (
                SELECT
                    NULLIF(TRIM(CAST({asin_col} AS VARCHAR)), '') AS asin,
                    NULLIF(TRIM(CAST({sql_column(present, "title")} AS VARCHAR)), '') AS title,
                    {author_expr} AS authors,
                    TRY_CAST(NULLIF(TRIM(CAST({sql_column(present, "average_rating")} AS VARCHAR)), '') AS DOUBLE) AS avg_rating,
                    TRY_CAST(NULLIF(TRIM(CAST({sql_column(present, "rating_number")} AS VARCHAR)), '') AS BIGINT) AS ratings_count,
                    COALESCE(CAST({categories} AS VARCHAR[]), []::VARCHAR[]) AS subjects
                FROM read_json(?, format='newline_delimited')
            )
            WHERE asin IS NOT NULL AND title IS NOT NULL
        )
        """

    def _ingest_books_sql(self, conn, present: Mapping[str, str]) -> int:  # noqa: ANN001
        """Insert ``books`` plus the ASIN ``aliases`` edge resolution needs.

        See :meth:`GoodreadsUCSDSource._ingest_books_sql` on why ``title_norm``
        is left to the store.
        """
        books_sql = f"""
        {self._books_cte(present)}
        INSERT INTO books (work_id, title, authors, asin, avg_rating, ratings_count, subjects)
        SELECT DISTINCT ON (work_id)
            work_id, title, authors, asin, avg_rating, ratings_count, subjects
        FROM final
        ON CONFLICT (work_id) DO UPDATE SET
            title = excluded.title,
            authors = excluded.authors,
            asin = excluded.asin,
            avg_rating = excluded.avg_rating,
            ratings_count = excluded.ratings_count,
            subjects = excluded.subjects
        """
        books_added = _count_of(conn.execute(books_sql, [str(self.path)]))

        aliases_sql = f"""
        {self._books_cte(present)}
        INSERT INTO aliases (id_type, id_value, work_id)
        SELECT DISTINCT ON (id_type, id_value) 'asin' AS id_type, asin AS id_value, work_id
        FROM final
        ON CONFLICT (id_type, id_value) DO UPDATE SET work_id = excluded.work_id
        """
        conn.execute(aliases_sql, [str(self.path)])
        return books_added

    def _ingest_edges_sql(self, conn, present: Mapping[str, str], *, max_rank: int) -> int:  # noqa: ANN001
        bought = _coalesce_lists(present, BOUGHT_KEYS)
        viewed = _coalesce_lists(present, VIEWED_KEYS)
        if bought is None and viewed is None:
            return 0
        # Derived from unresolved_ref() rather than hardcoded, so the SQL and
        # streaming paths cannot drift apart on the ref format.
        ref_prefix = unresolved_ref(self.name, "\x00").removesuffix("\x00")
        asin_col = sql_column(present, "parent_asin", fallback=sql_column(present, "asin"))
        edges_sql = f"""
        WITH kept AS (
            SELECT
                NULLIF(TRIM(CAST({asin_col} AS VARCHAR)), '') AS asin,
                {bought or "[]::VARCHAR[]"} AS bought,
                {viewed or "[]::VARCHAR[]"} AS viewed
            FROM read_json(?, format='newline_delimited')
            WHERE NULLIF(TRIM(CAST({asin_col} AS VARCHAR)), '') IS NOT NULL
              AND NULLIF(TRIM(CAST({sql_column(present, "title")} AS VARCHAR)), '') IS NOT NULL
        ),
        exploded AS (
            SELECT asin, '{EdgeKind.AZ_ALSO_BOUGHT.value}' AS kind,
                   NULLIF(TRIM(CAST(target AS VARCHAR)), '') AS target, position AS rank
            FROM kept, UNNEST(kept.bought) WITH ORDINALITY AS unnested(target, position)
            UNION ALL
            SELECT asin, '{EdgeKind.AZ_ALSO_VIEWED.value}' AS kind,
                   NULLIF(TRIM(CAST(target AS VARCHAR)), '') AS target, position AS rank
            FROM kept, UNNEST(kept.viewed) WITH ORDINALITY AS unnested(target, position)
        )
        INSERT INTO edges_raw (src, dst, kind, rank, source)
        SELECT DISTINCT ON (src, dst, kind, source) src, dst, kind, rank, source FROM (
            SELECT
                '{ref_prefix}' || asin AS src,
                '{ref_prefix}' || target AS dst,
                kind,
                rank,
                '{self.name.value}' AS source
            FROM exploded
            WHERE target IS NOT NULL AND target <> asin AND rank <= ?
        )
        ORDER BY rank
        ON CONFLICT (src, dst, kind, source) DO UPDATE SET rank = excluded.rank
        """
        return _count_of(conn.execute(edges_sql, [str(self.path), max_rank]))


def _author_names(value: object) -> tuple[str, ...]:
    """Pull author names out of the 2023 dump's nested ``author`` field.

    Shaped ``{"author": {"name": ...}}`` in the current release, but earlier
    exports use a bare string and some rows carry a list, so all three are
    accepted rather than crashing the ingest on the variant.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        name = text_or_none(value)
        return (name,) if name else ()
    if isinstance(value, dict):
        name = text_or_none(value.get("name"))
        return (name,) if name else ()
    if isinstance(value, list):
        names = []
        for entry in value:
            names.extend(_author_names(entry))
        return tuple(names)
    return ()


def _first_list(record: dict[str, object], keys: tuple[str, ...]) -> list[object]:
    """The first non-empty list among ``keys``, so key renames stay tolerable."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, list) and value:
            return value
    return []


def _coalesce_lists(present: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    """SQL mirror of :func:`_first_list`, or ``None`` if the dump has no such key.

    Only columns the dump actually contains are referenced -- a key absent from
    every record is not a column at all, and naming it is a bind error rather
    than a NULL.
    """
    parts = [
        f"NULLIF(COALESCE(CAST({key} AS VARCHAR[]), []::VARCHAR[]), []::VARCHAR[])"
        for key in keys
        if key in present
    ]
    if not parts:
        return None
    return f"COALESCE({', '.join(parts)}, []::VARCHAR[])"


def _reviews_field(value: str, label: str, coerce):  # noqa: ANN001, ANN201
    """Pull one labelled number out of a SNAP ``reviews:`` line.

    The line packs several ``label: number`` pairs onto one row
    ("total: 2  downloaded: 2  avg rating: 5"), so it is scanned for the label
    rather than split positionally.
    """
    marker = f"{label}:"
    index = value.find(marker)
    if index < 0:
        return None
    tail = value[index + len(marker) :].split()
    return coerce(tail[0]) if tail else None


def _count_of(result) -> int:  # noqa: ANN001
    """Rows affected by a DuckDB DML statement."""
    row = result.fetchone()
    return int(row[0]) if row else 0
