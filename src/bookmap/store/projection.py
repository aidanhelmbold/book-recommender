"""Project the stored graph into SciPy sparse matrices.

The fused edge table is pulled through Arrow so the endpoint columns arrive as
NumPy arrays without a Python-level row loop. At 20M edges that is the
difference between a sub-second build and a minute of cursor iteration on every
run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import scipy.sparse as sp

    from bookmap.store.db import Store

# Node ids are numbered in SQL and joined onto both endpoints there, so the
# string -> row mapping never crosses into Python per edge. Both statements order
# by ``node``, which is what makes ``idx`` here agree with the row order
# ``_NODE_SQL`` returns.
_EDGE_INDEX_SQL = """
WITH nodes AS (
    SELECT node, row_number() OVER (ORDER BY node) - 1 AS idx
    FROM (SELECT src AS node FROM edges_fused UNION SELECT dst FROM edges_fused)
)
SELECT s.idx AS i, d.idx AS j, e.weight AS w
FROM edges_fused e
JOIN nodes s ON s.node = e.src
JOIN nodes d ON d.node = e.dst
WHERE e.src <> e.dst
"""

_NODE_SQL = """
SELECT node FROM (
    SELECT src AS node FROM edges_fused UNION SELECT dst FROM edges_fused
) ORDER BY node
"""


@dataclass(frozen=True, slots=True)
class GraphProjection:
    """An immutable adjacency matrix plus the index that maps ids to rows."""

    adjacency: sp.csr_matrix
    """Symmetric, weighted, no self-loops."""

    work_ids: tuple[str, ...]
    """Row/column order; ``work_ids[i]`` is the id of row ``i``."""

    index: dict[str, int]

    @property
    def n_nodes(self) -> int:
        return len(self.work_ids)

    @property
    def n_edges(self) -> int:
        """Undirected edge count (each pair counted once)."""
        import scipy.sparse as sp

        # Strict upper triangle: the matrix is symmetric, so counting stored
        # entries would double every pair, and k=1 also ignores any diagonal.
        return int(sp.triu(self.adjacency, k=1).nnz)

    def degree(self) -> np.ndarray:
        """Unweighted degree per node, as a float array aligned to ``work_ids``."""
        # ``!= 0`` rather than a per-row nnz: a matrix assembled by summing
        # duplicate edges can hold explicitly stored zeros, which are not
        # neighbours.
        return np.asarray((self.adjacency != 0).sum(axis=1), dtype=float).ravel()

    def weighted_degree(self) -> np.ndarray:
        return np.asarray(self.adjacency.sum(axis=1), dtype=float).ravel()

    def row_of(self, work_id: str) -> int:
        """Row index for a work id. Raises KeyError if absent."""
        return self.index[work_id]

    def neighbors_of(self, work_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(neighbor_rows, weights)`` for a node."""
        row = self.index[work_id]
        adjacency = self.adjacency
        start, end = adjacency.indptr[row], adjacency.indptr[row + 1]
        cols = adjacency.indices[start:end]
        weights = adjacency.data[start:end]
        # Stored zeros are not edges; see :meth:`degree`.
        keep = weights != 0
        return cols[keep], weights[keep]

    def subgraph(self, work_ids: list[str]) -> GraphProjection:
        """Induced subgraph over the given nodes, preserving weights."""
        # Unknown ids are dropped rather than raising: callers pass community
        # members or PPR frontiers that may reference nodes trimmed from the
        # graph, and a missing node simply contributes no rows.
        kept: dict[str, int] = {}
        for work_id in work_ids:
            row = self.index.get(work_id)
            if row is not None and work_id not in kept:
                kept[work_id] = row

        rows = np.fromiter(kept.values(), dtype=np.int64, count=len(kept))
        # Two slices rather than np.ix_: fancy-indexing a CSR twice stays sparse,
        # while np.ix_ would build a dense selector.
        adjacency = self.adjacency[rows, :][:, rows].tocsr()
        names = tuple(kept)
        return GraphProjection(
            adjacency=adjacency,
            work_ids=names,
            index={name: i for i, name in enumerate(names)},
        )

    def to_networkx(self):  # noqa: ANN201 - networkx typing is loose
        """Convert to a NetworkX Graph. For tests and path-finding, not bulk work."""
        import networkx as nx

        graph = nx.from_scipy_sparse_array(self.adjacency, edge_attribute="weight")
        # from_scipy_sparse_array numbers nodes by row, so relabelling restores
        # the ids without losing isolated nodes.
        return nx.relabel_nodes(graph, dict(enumerate(self.work_ids)))


def _arrow_table(result):  # noqa: ANN001, ANN202 - duckdb/pyarrow types, imported lazily
    """Materialise a DuckDB result as an Arrow table across supported versions.

    DuckDB 1.5 renamed ``fetch_arrow_table`` to ``to_arrow_table`` and changed
    ``.arrow()`` to return a ``RecordBatchReader``; the project supports 1.4 too,
    so the accessor is chosen at runtime rather than pinned to one spelling.
    """
    fetch = getattr(result, "to_arrow_table", None) or result.fetch_arrow_table
    return fetch()


def project(store: Store) -> GraphProjection:
    """Build a projection from the store's fused edge table."""
    import scipy.sparse as sp

    conn = store.conn
    work_ids = tuple(_arrow_table(conn.execute(_NODE_SQL)).column("node").to_pylist())
    n_nodes = len(work_ids)

    edges = _arrow_table(conn.execute(_EDGE_INDEX_SQL))
    # Arrow -> NumPy in one buffer copy per column: the whole point of DuckDB
    # here is that no Python object is created per edge.
    i = np.asarray(edges.column("i").to_numpy(), dtype=np.int64)
    j = np.asarray(edges.column("j").to_numpy(), dtype=np.int64)
    weights = np.asarray(edges.column("w").to_numpy(), dtype=float)

    # ``edges_fused`` stores each undirected pair once with src < dst, so the
    # mirror image has to be added here to make the adjacency symmetric.
    adjacency = sp.coo_matrix(
        (
            np.concatenate([weights, weights]),
            (np.concatenate([i, j]), np.concatenate([j, i])),
        ),
        shape=(n_nodes, n_nodes),
        dtype=float,
    ).tocsr()
    adjacency.sum_duplicates()
    return GraphProjection(
        adjacency=adjacency,
        work_ids=work_ids,
        index={work_id: row for row, work_id in enumerate(work_ids)},
    )


def save_cache(projection: GraphProjection, path: str | Path) -> None:
    """Persist a projection to a ``.npz`` so repeated CLI calls skip the rebuild."""
    adjacency = projection.adjacency.tocsr()
    # The CSR buffers are stored raw rather than via scipy's save_npz so that
    # ``work_ids`` travels in the same file: row order is the only thing tying
    # cached scores back to books, and a separate file could drift from it.
    np.savez(
        Path(path),
        data=adjacency.data,
        indices=adjacency.indices,
        indptr=adjacency.indptr,
        shape=np.asarray(adjacency.shape, dtype=np.int64),
        work_ids=np.array(list(projection.work_ids), dtype="U"),
    )


def load_cache(path: str | Path) -> GraphProjection:
    """Load a projection previously written by :func:`save_cache`."""
    import scipy.sparse as sp

    with np.load(Path(path)) as cached:
        adjacency = sp.csr_matrix(
            (cached["data"], cached["indices"], cached["indptr"]),
            shape=tuple(int(size) for size in cached["shape"]),
        )
        # Back to plain ``str``: numpy scalars compare equal but leak their dtype
        # into anything that serialises these ids later.
        work_ids = tuple(str(work_id) for work_id in cached["work_ids"])
    return GraphProjection(
        adjacency=adjacency,
        work_ids=work_ids,
        index={work_id: row for row, work_id in enumerate(work_ids)},
    )
