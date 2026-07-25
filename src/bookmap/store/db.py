"""DuckDB-backed graph store.

DuckDB rather than SQLite because this workload is bulk-load plus full-table
analytical scan: ingest is a single ``read_json`` over a gzipped dump, fusion is
one ``GROUP BY`` over tens of millions of edges, and projection to SciPy hands
Arrow buffers straight to NumPy. Point lookups are slower than a B-tree, which
does not matter at typeahead latencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Self

from bookmap.models import Book, Edge, FusedEdge

if TYPE_CHECKING:  # pragma: no cover
    import duckdb


class Store:
    """Owns the DuckDB connection and all SQL.

    Opened read-only by the web app so that serving can never collide with an
    ingest (DuckDB permits a single writer).
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        raise NotImplementedError

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, *, read_only: bool = False) -> Self:
        """Open (creating if needed) and apply the schema."""
        raise NotImplementedError

    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def apply_schema(self) -> None:
        """Execute ``schema.sql``. Idempotent."""
        raise NotImplementedError

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        raise NotImplementedError

    # -- writes ------------------------------------------------------------

    def upsert_books(self, books: Iterable[Book], *, batch_size: int = 50_000) -> int:
        """Insert or update books, returning the number written.

        Must be idempotent: upserting the same book twice leaves one row.
        Populates ``title_norm`` and the ``aliases`` table as a side effect.
        """
        raise NotImplementedError

    def insert_raw_edges(self, edges: Iterable[Edge], *, batch_size: int = 100_000) -> int:
        """Insert directed edges, returning the number written.

        Must be idempotent on ``(src, dst, kind, source)``: re-ingesting a dump
        must not accumulate duplicate rows, since fusion sums over this table
        and duplicates would silently inflate edge weights.
        """
        raise NotImplementedError

    def replace_fused_edges(self, edges: Iterable[FusedEdge]) -> int:
        """Replace the whole fused graph. Returns rows written."""
        raise NotImplementedError

    def resolve_refs(self) -> int:
        """Rewrite namespaced edge endpoints onto canonical work ids.

        Adapters emit edges against ``source:raw_id`` refs because a dump
        routinely references a book before describing it. This runs after the
        book table is populated and rewrites both endpoints via ``aliases``,
        dropping edges whose endpoints never resolved. Returns edges dropped.
        """
        raise NotImplementedError

    def write_node_metrics(
        self,
        metrics: Iterable[tuple[str, int, float, int | None, float | None]],
    ) -> int:
        """Store ``(work_id, degree, weighted_degree, community, betweenness)``."""
        raise NotImplementedError

    def write_communities(self, labels: Iterable[tuple[int, str, int]]) -> int:
        """Store ``(community, label, size)``."""
        raise NotImplementedError

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
        raise NotImplementedError

    # -- reads -------------------------------------------------------------

    def get_book(self, work_id: str) -> Book | None:
        raise NotImplementedError

    def get_books(self, work_ids: Iterable[str]) -> dict[str, Book]:
        """Batch fetch; missing ids are simply absent from the result."""
        raise NotImplementedError

    def iter_books(self) -> Iterator[Book]:
        raise NotImplementedError

    def iter_raw_edges(self) -> Iterator[Edge]:
        raise NotImplementedError

    def iter_fused_edges(self) -> Iterator[FusedEdge]:
        raise NotImplementedError

    def search_titles(self, query: str, *, limit: int = 10) -> list[Book]:
        """Prefix/substring title search backing the web app's typeahead."""
        raise NotImplementedError

    def title_candidates(self) -> Iterator[tuple[str, str, tuple[str, ...]]]:
        """Yield ``(work_id, title, authors)`` for fuzzy seed resolution."""
        raise NotImplementedError

    def neighbors(self, work_id: str, *, limit: int = 25) -> list[tuple[str, float]]:
        """Fused neighbours of a node as ``(work_id, weight)``, strongest first."""
        raise NotImplementedError

    def node_community(self, work_id: str) -> int | None:
        raise NotImplementedError

    def community_labels(self) -> dict[int, str]:
        raise NotImplementedError

    def counts(self) -> dict[str, int]:
        """Row counts per table, for ``bookmap stats`` and test assertions."""
        raise NotImplementedError

    def alias_lookup(self, id_type: str, id_value: str) -> str | None:
        raise NotImplementedError
