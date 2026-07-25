"""DuckDB-backed graph store.

DuckDB rather than SQLite because this workload is bulk-load plus full-table
analytical scan: ingest is a single ``read_json`` over a gzipped dump, fusion is
one ``GROUP BY`` over tens of millions of edges, and projection to SciPy hands
Arrow buffers straight to NumPy. Point lookups are slower than a B-tree, which
does not matter at typeahead latencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Self

from bookmap.identity import normalize_title
from bookmap.models import Book, Edge, EdgeKind, FusedEdge, SourceName
from bookmap.sources.base import REF_ID_TYPE

if TYPE_CHECKING:  # pragma: no cover
    import duckdb

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

TABLES = (
    "books",
    "edges_raw",
    "edges_fused",
    "aliases",
    "node_metrics",
    "communities",
    "ingest_state",
)

# Column layouts, paired with an Arrow type code, for the bulk write path. The
# codes exist because an all-null Python column gives Arrow nothing to infer
# from, and a mistyped batch would be rejected by the insert rather than cast.
_BOOK_COLUMNS = (
    ("work_id", "str"),
    ("title", "str"),
    ("authors", "str[]"),
    ("year", "int"),
    ("isbn13", "str"),
    ("asin", "str"),
    ("goodreads_id", "str"),
    ("openlibrary_id", "str"),
    ("series", "str"),
    ("avg_rating", "float"),
    ("ratings_count", "int"),
    ("subjects", "str[]"),
    ("title_norm", "str"),
)
_ALIAS_COLUMNS = (("id_type", "str"), ("id_value", "str"), ("work_id", "str"))
_RAW_EDGE_COLUMNS = (
    ("src", "str"),
    ("dst", "str"),
    ("kind", "str"),
    ("rank", "int"),
    ("source", "str"),
)
_FUSED_EDGE_COLUMNS = (
    ("src", "str"),
    ("dst", "str"),
    ("weight", "float"),
    ("dir_asym", "float"),
    ("kinds", "str[]"),
)
_NODE_METRIC_COLUMNS = (
    ("work_id", "str"),
    ("degree", "int"),
    ("weighted_degree", "float"),
    ("community", "int"),
    ("betweenness", "float"),
)
_COMMUNITY_COLUMNS = (("community", "int"), ("label", "str"), ("size", "int"))
_INGEST_COLUMNS = (
    ("source", "str"),
    ("artifact", "str"),
    ("rows_done", "int"),
    ("books_added", "int"),
    ("edges_added", "int"),
    ("completed", "bool"),
)

_BOOK_SELECT = (
    "work_id, title, authors, year, isbn13, asin, goodreads_id, "
    "openlibrary_id, series, avg_rating, ratings_count, subjects"
)

# Which identifiers earn an alias row. Every one of these is a key some other
# source will use to point at this work.
_ALIAS_FIELDS = ("isbn13", "asin", "goodreads_id", "openlibrary_id")

_STAGING_VIEW = "_bookmap_batch"


class Store:
    """Owns the DuckDB connection and all SQL.

    Opened read-only by the web app so that serving can never collide with an
    ingest (DuckDB permits a single writer).
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        import duckdb

        self.path = Path(path)
        self.read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = duckdb.connect(
            str(self.path), read_only=read_only
        )

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, *, read_only: bool = False) -> Self:
        """Open (creating if needed) and apply the schema."""
        store = cls(path, read_only=read_only)
        if not read_only:
            # DDL is refused on a read-only database, and a reader has by
            # definition nothing to migrate: whoever wrote the file applied it.
            store.apply_schema()
        return store

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def apply_schema(self) -> None:
        """Execute ``schema.sql``. Idempotent."""
        self.conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError(f"store is closed: {self.path}")
        return self._conn

    def __repr__(self) -> str:
        mode = "read-only" if self.read_only else "read-write"
        return f"Store({str(self.path)!r}, {mode})"

    # -- bulk write plumbing -----------------------------------------------

    def _upsert(
        self,
        table: str,
        columns: Sequence[tuple[str, str]],
        rows: Sequence[tuple[object, ...]],
        *,
        key_width: int,
        extra_sql: str = "",
    ) -> int:
        """Upsert one batch through a registered Arrow table.

        One statement per batch, not one per row: DuckDB's ``executemany`` runs
        each insert as its own transaction, which measures in minutes for a
        200k-row batch that Arrow loads in under a second.

        ``key_width`` is how many leading columns form the primary key; rows are
        de-duplicated on it because a single ``INSERT OR REPLACE`` cannot resolve
        two conflicting versions of the same key in one statement.
        """
        if not rows:
            return 0

        deduped = {row[:key_width]: row for row in rows}  # last write wins
        batch = _arrow_batch(columns, list(deduped.values()))
        names = ", ".join(name for name, _ in columns)
        self.conn.register(_STAGING_VIEW, batch)
        try:
            self.conn.execute(
                f"INSERT OR REPLACE INTO {table} ({names}{extra_sql and ', ' + extra_sql})"  # noqa: S608 - table/column names are module constants
                f" SELECT {names}{extra_sql and ', ' + extra_sql} FROM {_STAGING_VIEW}"
            )
        finally:
            self.conn.unregister(_STAGING_VIEW)
        return len(deduped)

    # -- writes ------------------------------------------------------------

    def upsert_books(self, books: Iterable[Book], *, batch_size: int = 50_000) -> int:
        """Insert or update books, returning the number written.

        Must be idempotent: upserting the same book twice leaves one row.
        Populates ``title_norm`` and the ``aliases`` table as a side effect.
        """
        written = 0
        for batch in _batched(books, batch_size):
            rows = []
            aliases = []
            for book in batch:
                rows.append(
                    (
                        book.work_id,
                        book.title,
                        list(book.authors),
                        book.year,
                        book.isbn13,
                        book.asin,
                        book.goodreads_id,
                        book.openlibrary_id,
                        book.series,
                        book.avg_rating,
                        book.ratings_count,
                        list(book.subjects),
                        # Denormalised at write time: seed resolution and the
                        # typeahead both scan this column on every query.
                        normalize_title(book.title),
                    )
                )
                for field in _ALIAS_FIELDS:
                    value = getattr(book, field)
                    if value:
                        aliases.append((field, value, book.work_id))

            written += self._upsert("books", _BOOK_COLUMNS, rows, key_width=1)
            # Without these, edges emitted against upstream identifiers can
            # never be resolved onto this node.
            self._upsert("aliases", _ALIAS_COLUMNS, aliases, key_width=2)
        return written

    def insert_raw_edges(self, edges: Iterable[Edge], *, batch_size: int = 100_000) -> int:
        """Insert directed edges, returning the number written.

        Must be idempotent on ``(src, dst, kind, source)``: re-ingesting a dump
        must not accumulate duplicate rows, since fusion sums over this table
        and duplicates would silently inflate edge weights.
        """
        written = 0
        for batch in _batched(edges, batch_size):
            rows = [
                (edge.src, edge.dst, str(edge.kind), edge.rank, str(edge.source))
                for edge in batch
            ]
            # The key is (src, dst, kind, source); rank is the payload, so the
            # same pair attested by two sources stays two rows.
            written += self._upsert(
                "edges_raw", _RAW_EDGE_COLUMNS, _key_reordered(rows), key_width=4
            )
        return written

    def replace_fused_edges(self, edges: Iterable[FusedEdge]) -> int:
        """Replace the whole fused graph. Returns rows written."""
        self.conn.execute("DELETE FROM edges_fused")
        for batch in _batched(edges, 100_000):
            rows = [
                (
                    edge.src,
                    edge.dst,
                    edge.weight,
                    edge.dir_asym,
                    [str(kind) for kind in edge.kinds],
                )
                for edge in batch
            ]
            self._upsert("edges_fused", _FUSED_EDGE_COLUMNS, rows, key_width=2)
        # Counted from the table rather than from the batches so a pair repeated
        # across two batches is reported once, as it is stored once.
        return int(self.conn.execute("SELECT count(*) FROM edges_fused").fetchone()[0])

    def resolve_refs(self) -> int:
        """Rewrite namespaced edge endpoints onto canonical work ids.

        Adapters emit edges against ``source:raw_id`` refs because a dump
        routinely references a book before describing it. This runs after the
        book table is populated and rewrites both endpoints via ``aliases``,
        dropping edges whose endpoints never resolved. Returns edges dropped.
        """
        conn = self.conn
        before = int(conn.execute("SELECT count(*) FROM edges_raw").fetchone()[0])
        if not before:
            return 0

        # A VALUES join supplies the ref-prefix -> alias id_type mapping, so the
        # store and the adapters share one definition of it.
        prefixes = ", ".join(["(?, ?)"] * len(REF_ID_TYPE))
        params: list[object] = []
        for source, id_type in REF_ID_TYPE.items():
            params.extend([str(source), id_type])

        # An endpoint resolves to itself when it carries no known ref prefix --
        # which is what makes a second pass a no-op -- and to NULL when it is a
        # ref whose alias is absent, which is what marks it for dropping.
        conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE _bookmap_resolved AS
            WITH ref_types(prefix, id_type) AS (VALUES {prefixes})
            SELECT
                COALESCE(sa.work_id, CASE WHEN st.prefix IS NULL THEN e.src END) AS src,
                COALESCE(da.work_id, CASE WHEN dt.prefix IS NULL THEN e.dst END) AS dst,
                e.kind AS kind,
                e."rank" AS "rank",
                e.source AS source
            FROM edges_raw e
            LEFT JOIN ref_types st
                   ON strpos(e.src, ':') > 0 AND st.prefix = split_part(e.src, ':', 1)
            LEFT JOIN aliases sa
                   ON sa.id_type = st.id_type
                  AND sa.id_value = substr(e.src, strpos(e.src, ':') + 1)
            LEFT JOIN ref_types dt
                   ON strpos(e.dst, ':') > 0 AND dt.prefix = split_part(e.dst, ':', 1)
            LEFT JOIN aliases da
                   ON da.id_type = dt.id_type
                  AND da.id_value = substr(e.dst, strpos(e.dst, ':') + 1)
            """,  # noqa: S608 - the only interpolation is a placeholder list
            params,
        )
        try:
            kept = int(
                conn.execute(
                    "SELECT count(*) FROM _bookmap_resolved "
                    "WHERE src IS NOT NULL AND dst IS NOT NULL AND src <> dst"
                ).fetchone()[0]
            )
            conn.execute("DELETE FROM edges_raw")
            # Two refs can resolve onto one work (two editions of it), which both
            # collapses a pair into a self-loop and collides on the primary key;
            # the strongest surviving rank wins.
            conn.execute(
                """
                INSERT INTO edges_raw (src, dst, kind, "rank", source)
                SELECT src, dst, kind, min("rank"), source
                FROM _bookmap_resolved
                WHERE src IS NOT NULL AND dst IS NOT NULL AND src <> dst
                GROUP BY src, dst, kind, source
                """
            )
        finally:
            conn.execute("DROP TABLE IF EXISTS _bookmap_resolved")
        return before - kept

    def write_node_metrics(
        self,
        metrics: Iterable[tuple[str, int, float, int | None, float | None]],
    ) -> int:
        """Store ``(work_id, degree, weighted_degree, community, betweenness)``."""
        written = 0
        for batch in _batched(metrics, 100_000):
            written += self._upsert(
                "node_metrics", _NODE_METRIC_COLUMNS, list(batch), key_width=1
            )
        return written

    def write_communities(self, labels: Iterable[tuple[int, str, int]]) -> int:
        """Store ``(community, label, size)``."""
        return self._upsert("communities", _COMMUNITY_COLUMNS, list(labels), key_width=1)

    def mark_ingest(
        self,
        source: str,
        artifact: str,
        *,
        rows_done: int = 0,
        books_added: int = 0,
        edges_added: int = 0,
        completed: bool = False,
    ) -> None:
        """Record ingest progress so a large dump can be resumed."""
        # One row per (source, artifact), refreshed in place: a resuming ingest
        # asks this table where it got to, and two rows have no answer.
        self._upsert(
            "ingest_state",
            _INGEST_COLUMNS,
            [(str(source), str(artifact), rows_done, books_added, edges_added, completed)],
            key_width=2,
            extra_sql="current_timestamp AS updated_at",
        )

    # -- reads -------------------------------------------------------------

    def get_book(self, work_id: str) -> Book | None:
        row = self.conn.execute(
            f"SELECT {_BOOK_SELECT} FROM books WHERE work_id = ?",  # noqa: S608 - column list is a module constant
            [work_id],
        ).fetchone()
        return _book_from_row(row) if row else None

    def get_books(self, work_ids: Iterable[str]) -> dict[str, Book]:
        """Batch fetch; missing ids are simply absent from the result."""
        ids = list(work_ids)
        if not ids:
            return {}
        # A list parameter keeps this one round trip regardless of batch size.
        rows = self.conn.execute(
            f"SELECT {_BOOK_SELECT} FROM books WHERE work_id = ANY(?)",  # noqa: S608 - column list is a module constant
            [ids],
        ).fetchall()
        return {row[0]: _book_from_row(row) for row in rows}

    def iter_books(self) -> Iterator[Book]:
        yield from (
            _book_from_row(row)
            for row in self._stream(f"SELECT {_BOOK_SELECT} FROM books")  # noqa: S608 - column list is a module constant
        )

    def iter_raw_edges(self) -> Iterator[Edge]:
        for src, dst, kind, rank, source in self._stream(
            'SELECT src, dst, kind, "rank", source FROM edges_raw'
        ):
            yield Edge(
                src=src, dst=dst, kind=EdgeKind(kind), rank=rank, source=SourceName(source)
            )

    def iter_fused_edges(self) -> Iterator[FusedEdge]:
        for src, dst, weight, dir_asym, kinds in self._stream(
            "SELECT src, dst, weight, dir_asym, kinds FROM edges_fused"
        ):
            yield FusedEdge(
                src=src,
                dst=dst,
                weight=weight,
                dir_asym=dir_asym,
                kinds=tuple(EdgeKind(kind) for kind in kinds or ()),
            )

    def search_titles(self, query: str, *, limit: int = 10) -> list[Book]:
        """Prefix/substring title search backing the web app's typeahead."""
        needle = _like_needle(query)
        if not needle:
            return []
        # Both columns: ``title_norm`` has leading articles stripped, so a user
        # typing "the hobbit" only matches through the raw title.
        rows = self.conn.execute(
            f"""
            SELECT {_BOOK_SELECT} FROM books
            WHERE title_norm LIKE ? OR lower(title) LIKE ?
            ORDER BY title_norm LIKE ? DESC,
                     ratings_count DESC NULLS LAST,
                     length(title),
                     work_id
            LIMIT ?
            """,  # noqa: S608 - column list is a module constant
            [f"%{needle}%", f"%{needle}%", f"{needle}%", limit],
        ).fetchall()
        return [_book_from_row(row) for row in rows]

    def title_candidates(self) -> Iterator[tuple[str, str, tuple[str, ...]]]:
        """Yield ``(work_id, title, authors)`` for fuzzy seed resolution."""
        for work_id, title, authors in self._stream("SELECT work_id, title, authors FROM books"):
            yield work_id, title, tuple(authors or ())

    def neighbors(self, work_id: str, *, limit: int = 25) -> list[tuple[str, float]]:
        """Fused neighbours of a node as ``(work_id, weight)``, strongest first."""
        # Fused edges are stored once with src < dst, so a node's neighbourhood
        # lives in both columns.
        rows = self.conn.execute(
            """
            SELECT neighbor, weight FROM (
                SELECT dst AS neighbor, weight FROM edges_fused WHERE src = ?
                UNION ALL
                SELECT src AS neighbor, weight FROM edges_fused WHERE dst = ?
            )
            ORDER BY weight DESC, neighbor
            LIMIT ?
            """,
            [work_id, work_id, limit],
        ).fetchall()
        return [(neighbor, weight) for neighbor, weight in rows]

    def node_community(self, work_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT community FROM node_metrics WHERE work_id = ?", [work_id]
        ).fetchone()
        return row[0] if row else None

    def community_labels(self) -> dict[int, str]:
        return {
            community: label
            for community, label in self.conn.execute(
                "SELECT community, label FROM communities"
            ).fetchall()
        }

    def counts(self) -> dict[str, int]:
        """Row counts per table, for ``bookmap stats`` and test assertions."""
        # One query rather than seven round trips.
        selects = ", ".join(f"(SELECT count(*) FROM {table})" for table in TABLES)
        row = self.conn.execute(f"SELECT {selects}").fetchone()  # noqa: S608 - table names are module constants
        return dict(zip(TABLES, (int(value) for value in row), strict=True))

    def alias_lookup(self, id_type: str, id_value: str) -> str | None:
        row = self.conn.execute(
            "SELECT work_id FROM aliases WHERE id_type = ? AND id_value = ?",
            [id_type, id_value],
        ).fetchone()
        return row[0] if row else None

    # -- read plumbing -----------------------------------------------------

    def _stream(self, sql: str, batch_size: int = 10_000) -> Iterator[tuple[object, ...]]:
        """Stream a result set on its own cursor.

        A cursor rather than the shared connection because callers legitimately
        write while iterating (``replace_fused_edges(fuse_edges(iter_raw_edges()))``),
        and a single connection holds only one result set at a time.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql)
            while rows := cursor.fetchmany(batch_size):
                yield from rows
        finally:
            cursor.close()


def _arrow_batch(columns: Sequence[tuple[str, str]], rows: Sequence[tuple[object, ...]]):  # noqa: ANN202 - pyarrow.Table, imported lazily
    """Turn row tuples into a typed Arrow table for the bulk insert path."""
    import pyarrow as pa

    types = {
        "str": pa.string(),
        "int": pa.int64(),
        "float": pa.float64(),
        "bool": pa.bool_(),
        "str[]": pa.list_(pa.string()),
    }
    return pa.table(
        {
            name: pa.array([row[position] for row in rows], type=types[code])
            for position, (name, code) in enumerate(columns)
        }
    )


def _key_reordered(rows: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
    """Move ``rank`` behind ``source`` so the primary key is a row prefix.

    ``edges_raw`` is keyed on (src, dst, kind, source) but stores rank in the
    middle; de-duplication in :meth:`Store._upsert` slices a key off the front.
    """
    return [(src, dst, kind, source, rank) for src, dst, kind, rank, source in rows]


def _batched(items: Iterable[object], size: int) -> Iterator[list[object]]:
    """Chunk a stream. Dumps are far larger than memory, so nothing is listed."""
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _book_from_row(row: Sequence[object]) -> Book:
    return Book(
        work_id=row[0],
        title=row[1],
        authors=tuple(row[2] or ()),
        year=row[3],
        isbn13=row[4],
        asin=row[5],
        goodreads_id=row[6],
        openlibrary_id=row[7],
        series=row[8],
        avg_rating=row[9],
        ratings_count=row[10],
        subjects=tuple(row[11] or ()),
    )


def _like_needle(query: str) -> str:
    """Reduce a query to alphanumerics for LIKE.

    Dropping punctuation also removes ``%`` and ``_``, so a user's query cannot
    turn into a wildcard.
    """
    stripped = "".join(ch if ch.isalnum() else " " for ch in query.lower())
    return " ".join(stripped.split())
