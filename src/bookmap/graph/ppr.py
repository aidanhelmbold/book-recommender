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

import numpy as np
import scipy.sparse as sp


def row_normalize(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalise a weighted adjacency matrix into a transition matrix.

    Zero-degree rows are left as zero rows; :func:`personalized_pagerank`
    redistributes their mass so probability is conserved.
    """
    adjacency = sp.csr_matrix(adjacency)
    strength = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros(strength.shape[0], dtype=float)
    nonzero = strength > 0
    inverse[nonzero] = 1.0 / strength[nonzero]
    # Left-multiplying by a diagonal scales each row and stays sparse throughout.
    return sp.diags(inverse).dot(adjacency).tocsr()


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
    adjacency = sp.csr_matrix(adjacency)
    n_nodes = adjacency.shape[0]

    rows = np.asarray(seed_rows, dtype=np.intp).ravel()
    if rows.size == 0:
        raise ValueError("personalized_pagerank requires at least one seed row")
    if n_nodes == 0:
        raise ValueError("cannot rank an empty graph")
    if rows.min() < 0 or rows.max() >= n_nodes:
        raise ValueError(f"seed rows out of range for a graph of {n_nodes} nodes")

    if seed_weights is None:
        weights = np.ones(rows.size, dtype=float)
    else:
        weights = np.asarray(seed_weights, dtype=float).ravel()
        if weights.size != rows.size:
            raise ValueError("seed_weights must be the same length as seed_rows")
        if (weights < 0).any():
            raise ValueError("seed_weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("seed_weights must sum to a positive value")

    # The restart distribution s. It doubles as the dangling distribution below.
    restart = np.zeros(n_nodes, dtype=float)
    np.add.at(restart, rows, weights / total)

    transition = row_normalize(adjacency)
    # Transpose once up front: the iteration needs p^T M, and repeatedly doing
    # matvec against a CSC matrix is slower than a single conversion.
    transition_t = transition.transpose().tocsr()
    dangling = np.asarray(adjacency.sum(axis=1)).ravel() <= 0

    damping = 1.0 - alpha
    p = restart.copy()
    for _ in range(max_iter):
        # Dangling rows have no outgoing mass, so theirs is teleported along the
        # seed vector. Skipping this leaks probability and the result stops
        # summing to 1 -- NetworkX does the same when ``dangling=None``.
        leaked = float(p[dangling].sum())
        nxt = damping * (transition_t.dot(p) + leaked * restart) + alpha * restart
        converged = np.abs(nxt - p).sum() < n_nodes * tol
        p = nxt
        if converged:
            break

    # Clip sub-epsilon negatives from rounding and renormalise, so callers can
    # rely on this being an honest probability distribution.
    np.clip(p, 0.0, None, out=p)
    mass = p.sum()
    if mass > 0:
        p /= mass
    return p


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
    ppr = np.asarray(ppr, dtype=float)
    degree = np.asarray(degree, dtype=float)
    if beta == 0.0:
        return ppr.copy()

    positive = degree > 0
    # Substituting 1.0 in the divisor keeps the power fully vectorised without
    # tripping a divide-by-zero warning; where() discards those entries anyway.
    divisor = np.where(positive, degree, 1.0) ** beta
    return np.where(positive, ppr / divisor, 0.0)
