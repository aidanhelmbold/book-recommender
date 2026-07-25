"""Test-only helpers: independent oracles and hand-built graphs.

Nothing here imports bookmap implementation code beyond the frozen contracts, so
these helpers stay a genuine external check rather than a mirror of the code
under test.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np


def adjusted_rand_index(labels_true: list[int] | np.ndarray, labels_pred: list[int] | np.ndarray) -> float:
    """Adjusted Rand Index, implemented here to avoid a scikit-learn dependency.

    Measures partition agreement corrected for chance: 1.0 is identical, ~0.0 is
    random. Used to assert community detection recovers a planted partition.
    """
    labels_true = list(labels_true)
    labels_pred = list(labels_pred)
    if len(labels_true) != len(labels_pred):
        raise ValueError("label arrays must be the same length")
    n = len(labels_true)
    if n == 0:
        return 1.0

    contingency = Counter(zip(labels_true, labels_pred, strict=True))
    a_counts = Counter(labels_true)
    b_counts = Counter(labels_pred)

    index = sum(comb(v, 2) for v in contingency.values())
    sum_a = sum(comb(v, 2) for v in a_counts.values())
    sum_b = sum(comb(v, 2) for v in b_counts.values())
    total = comb(n, 2)
    if total == 0:
        return 1.0
    expected = sum_a * sum_b / total
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def csr_from_pairs(
    pairs: list[tuple[str, str, float]],
    nodes: list[str] | None = None,
):
    """Build a symmetric CSR adjacency matrix plus its node index.

    Returns ``(adjacency, work_ids, index)`` -- the same triple a
    ``GraphProjection`` carries, so tests can construct graphs without going
    through the store.
    """
    import scipy.sparse as sp

    if nodes is None:
        seen: dict[str, None] = {}
        for u, v, _ in pairs:
            seen.setdefault(u, None)
            seen.setdefault(v, None)
        nodes = list(seen)
    index = {name: i for i, name in enumerate(nodes)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u, v, w in pairs:
        i, j = index[u], index[v]
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    adjacency = sp.csr_matrix(
        (data, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=float
    )
    adjacency.sum_duplicates()
    return adjacency, tuple(nodes), index


def make_projection(pairs: list[tuple[str, str, float]], nodes: list[str] | None = None):
    """Build a real ``GraphProjection`` from hand-specified edges."""
    from bookmap.store.projection import GraphProjection

    adjacency, work_ids, index = csr_from_pairs(pairs, nodes)
    return GraphProjection(adjacency=adjacency, work_ids=work_ids, index=index)


def networkx_ppr(
    pairs: list[tuple[str, str, float]],
    seeds: list[str],
    *,
    restart_alpha: float = 0.15,
) -> dict[str, float]:
    """Reference personalized PageRank via NetworkX.

    The independent oracle for :func:`bookmap.graph.ppr.personalized_pagerank`.
    Note the convention shift: NetworkX's ``alpha`` is the *damping* factor,
    which is ``1 - restart_alpha``.
    """
    import networkx as nx

    graph = nx.Graph()
    for u, v, w in pairs:
        graph.add_edge(u, v, weight=w)
    personalization = {node: 0.0 for node in graph.nodes}
    for seed in seeds:
        personalization[seed] = 1.0 / len(seeds)
    return nx.pagerank(
        graph,
        alpha=1.0 - restart_alpha,
        personalization=personalization,
        weight="weight",
        tol=1e-12,
        max_iter=500,
    )


def planted_partition(
    n_blocks: int = 4,
    block_size: int = 25,
    p_in: float = 0.35,
    p_out: float = 0.01,
    seed: int = 42,
) -> tuple[list[tuple[str, str, float]], list[int]]:
    """A graph with known community structure.

    Returns ``(pairs, ground_truth_labels)``. Dense inside blocks, sparse
    between them -- community detection that cannot recover this cannot recover
    genre clusters either.
    """
    rng = np.random.default_rng(seed)
    labels = [i // block_size for i in range(n_blocks * block_size)]
    names = [f"n{i}" for i in range(len(labels))]
    pairs: list[tuple[str, str, float]] = []
    for i, j in combinations(range(len(labels)), 2):
        p = p_in if labels[i] == labels[j] else p_out
        if rng.random() < p:
            pairs.append((names[i], names[j], 1.0))
    return pairs, labels


def jaccard(a: set[str], b: set[str]) -> float:
    """Set Jaccard similarity; the oracle for MMR's redundancy term."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
