"""Centrality measures, including the ones that make the map interesting.

Degree is needed by hub damping. Betweenness is needed for *bridge books* --
titles that sit between two otherwise-separate reading communities. Those are
often the most useful recommendation of all for someone trying to move from one
genre into another, and they are invisible to plain similarity ranking.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.projection import GraphProjection


def degrees(projection: GraphProjection) -> np.ndarray:
    """Unweighted degree per node, aligned to ``projection.work_ids``."""
    adjacency = sp.csr_matrix(projection.adjacency)
    # Count structural nonzeros rather than getnnz(): a projection assembled by
    # summing duplicate edges can carry explicitly stored zeros, which are not
    # neighbours.
    return np.asarray((adjacency != 0).sum(axis=1), dtype=float).ravel()


def weighted_degrees(projection: GraphProjection) -> np.ndarray:
    adjacency = sp.csr_matrix(projection.adjacency)
    return np.asarray(adjacency.sum(axis=1), dtype=float).ravel()


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
    adjacency = sp.csr_matrix(projection.adjacency)
    n_nodes = adjacency.shape[0]
    betweenness = np.zeros(n_nodes, dtype=float)
    if n_nodes <= 2:
        return betweenness

    indices = adjacency.indices
    indptr = adjacency.indptr

    exact = k_samples >= n_nodes
    if exact:
        pivots = np.arange(n_nodes)
    else:
        rng = np.random.default_rng(seed)
        # Sorted so the accumulation order -- and therefore the floating-point
        # result -- is fixed by the seed alone.
        pivots = np.sort(rng.choice(n_nodes, size=max(k_samples, 1), replace=False))

    for source in pivots.tolist():
        _brandes_accumulate(source, indptr, indices, betweenness)

    # Match NetworkX's normalisation exactly: the pair count for a directed
    # traversal of an undirected graph, which already double-counts each pair.
    scale = 1.0 / ((n_nodes - 1) * (n_nodes - 2))
    if not exact:
        # Extrapolate from the sampled pivots to all n of them.
        scale *= n_nodes / pivots.shape[0]
    betweenness *= scale
    return betweenness


def _brandes_accumulate(
    source: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    betweenness: np.ndarray,
) -> None:
    """One Brandes single-source pass: BFS forward, dependency sum backward.

    Unweighted (BFS) shortest paths, matching
    ``networkx.betweenness_centrality`` with its default ``weight=None``: fused
    weights measure similarity, not distance, so treating them as edge lengths
    would invert their meaning.
    """
    n_nodes = betweenness.shape[0]
    sigma = np.zeros(n_nodes, dtype=float)
    distance = np.full(n_nodes, -1, dtype=np.int64)
    predecessors: list[list[int]] = [[] for _ in range(n_nodes)]

    sigma[source] = 1.0
    distance[source] = 0
    stack: list[int] = []
    queue: deque[int] = deque([source])

    while queue:
        node = queue.popleft()
        stack.append(node)
        node_distance = distance[node]
        node_sigma = sigma[node]
        for neighbor in indices[indptr[node] : indptr[node + 1]].tolist():
            if distance[neighbor] < 0:
                distance[neighbor] = node_distance + 1
                queue.append(neighbor)
            if distance[neighbor] == node_distance + 1:
                sigma[neighbor] += node_sigma
                predecessors[neighbor].append(node)

    delta = np.zeros(n_nodes, dtype=float)
    while stack:
        node = stack.pop()
        coefficient = (1.0 + delta[node]) / sigma[node]
        for predecessor in predecessors[node]:
            delta[predecessor] += sigma[predecessor] * coefficient
        if node != source:
            betweenness[node] += delta[node]


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
    adjacency = sp.csr_matrix(projection.adjacency)
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return []

    labels = np.asarray(labels).ravel()
    if labels.shape[0] != n_nodes:
        raise ValueError("labels must be aligned to the projection's nodes")

    indices = adjacency.indices
    indptr = adjacency.indptr
    data = adjacency.data

    ranked: list[tuple[str, float, tuple[int, ...]]] = []
    for row in range(n_nodes):
        start, end = indptr[row], indptr[row + 1]
        neighbor_labels = labels[indices[start:end]]
        weights = data[start:end]
        total = float(weights.sum())
        if total <= 0:
            continue

        own = int(labels[row])
        crossing = neighbor_labels != own
        cross_weight = float(weights[crossing].sum())
        touched = tuple(sorted({own, *(int(x) for x in neighbor_labels.tolist())}))

        # Fraction of a book's connection strength that leaves its own cluster,
        # scaled by the breadth of the crossing: a title straddling three genres
        # is a more useful gateway than one straddling two.
        score = (cross_weight / total) * len(touched)
        if score <= 0:
            continue
        ranked.append((projection.work_ids[row], score, touched))

    # work_id tie-break so equally-good bridges come back in a stable order.
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:top_n]
