"""Centrality measures, including the ones that make the map interesting.

Degree is needed by hub damping. Betweenness is needed for *bridge books* --
titles that sit between two otherwise-separate reading communities. Those are
often the most useful recommendation of all for someone trying to move from one
genre into another, and they are invisible to plain similarity ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.projection import GraphProjection


def degrees(projection: GraphProjection) -> np.ndarray:
    """Unweighted degree per node, aligned to ``projection.work_ids``."""
    raise NotImplementedError


def weighted_degrees(projection: GraphProjection) -> np.ndarray:
    raise NotImplementedError


def approximate_betweenness(
    projection: GraphProjection,
    *,
    k_samples: int = 500,
    seed: int = 1917,
) -> np.ndarray:
    """Sampled betweenness centrality.

    Exact betweenness is O(VE) and hopeless at millions of nodes, so pivots are
    sampled. ``k_samples >= n_nodes`` computes it exactly, which is what the
    tests use to check the approximation is unbiased in the limit.
    """
    raise NotImplementedError


def bridge_books(
    projection: GraphProjection,
    labels: np.ndarray,
    *,
    top_n: int = 20,
) -> list[tuple[str, float, tuple[int, ...]]]:
    """Find books linking distinct communities.

    Scores each node by how much of its edge weight crosses community
    boundaries, weighted by how many distinct communities it touches. Returns
    ``(work_id, score, communities_touched)`` for the top ``top_n``.
    """
    raise NotImplementedError
