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
        raise NotImplementedError

    @property
    def n_edges(self) -> int:
        """Undirected edge count (each pair counted once)."""
        raise NotImplementedError

    def degree(self) -> np.ndarray:
        """Unweighted degree per node, as a float array aligned to ``work_ids``."""
        raise NotImplementedError

    def weighted_degree(self) -> np.ndarray:
        raise NotImplementedError

    def row_of(self, work_id: str) -> int:
        """Row index for a work id. Raises KeyError if absent."""
        raise NotImplementedError

    def neighbors_of(self, work_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(neighbor_rows, weights)`` for a node."""
        raise NotImplementedError

    def subgraph(self, work_ids: list[str]) -> GraphProjection:
        """Induced subgraph over the given nodes, preserving weights."""
        raise NotImplementedError

    def to_networkx(self):  # noqa: ANN201 - networkx typing is loose
        """Convert to a NetworkX Graph. For tests and path-finding, not bulk work."""
        raise NotImplementedError


def project(store: Store) -> GraphProjection:
    """Build a projection from the store's fused edge table."""
    raise NotImplementedError


def save_cache(projection: GraphProjection, path: str | Path) -> None:
    """Persist a projection to a ``.npz`` so repeated CLI calls skip the rebuild."""
    raise NotImplementedError


def load_cache(path: str | Path) -> GraphProjection:
    """Load a projection previously written by :func:`save_cache`."""
    raise NotImplementedError
