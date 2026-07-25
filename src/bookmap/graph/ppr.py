"""Personalized PageRank (random walk with restart).

This is the heart of the recommender. A plain "most similar to each seed" query
answers the wrong question -- the user supplies a *set* of books and wants what
suits the set as a whole. Restarting a random walk uniformly over the seed set
does exactly that, and rewards books reachable from several seeds at once over
books strongly tied to just one.

Implemented as sparse power iteration rather than via NetworkX because the real
graph has millions of nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import scipy.sparse as sp


def row_normalize(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalise a weighted adjacency matrix into a transition matrix.

    Zero-degree rows are left as zero rows; :func:`personalized_pagerank`
    redistributes their mass so probability is conserved.
    """
    raise NotImplementedError


def personalized_pagerank(
    adjacency: sp.csr_matrix,
    seed_rows: np.ndarray | list[int],
    *,
    alpha: float = 0.15,
    tol: float = 1e-8,
    max_iter: int = 200,
    seed_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the stationary distribution of a walk restarting on the seeds.

    ``alpha`` is the restart probability (so the PageRank damping factor is
    ``1 - alpha``). ``seed_weights`` allows uneven seed emphasis; the default is
    uniform over ``seed_rows``.

    The result is a probability distribution: non-negative and summing to 1,
    including in the presence of dangling nodes.
    """
    raise NotImplementedError


def hub_damped_scores(
    ppr: np.ndarray,
    degree: np.ndarray,
    *,
    beta: float = 0.25,
) -> np.ndarray:
    """Divide PPR mass by ``degree ** beta`` to suppress bestseller hubs.

    Without this, high-degree nodes win for every seed set -- a book connected
    to 4000 others collects walk probability regardless of relevance, so
    undamped PPR recommends the same famous titles to everyone. ``beta`` trades
    popularity against specificity; 0 disables damping entirely.

    Zero-degree nodes keep their (zero) score rather than dividing by zero.
    """
    raise NotImplementedError
