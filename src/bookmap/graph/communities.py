"""Community detection and cluster naming.

Communities are what turn a recommendation list into a *map*: they separate the
"hard SF" region from the "literary fiction" region, which drives both the map's
colouring and the diversity reranking (a good result set spans the clusters the
seeds touch rather than piling into the densest one).

Leiden via python-igraph where available -- it scales to millions of nodes and,
unlike classic Louvain, guarantees internally connected communities. NetworkX's
greedy modularity is the fallback for small graphs and tests.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.projection import GraphProjection

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for from by with without into onto over
    under about as is are was were be been being am do does did doing have has had
    it its it's this that these those there here then than so such no not nor only
    own same too very can will just should now i you he she they we me him her them
    my your his their our who whom which what when where why how all any both each
    few more most other some s t don't book books vol volume edition new
    """.split()
)
"""Deliberately generic English filler plus a few bibliographic words that carry
no genre signal. Domain filler like "novel" is left to the IDF term to suppress,
which is what makes labels adapt to whatever corpus is loaded."""


def detect_communities(
    projection: GraphProjection,
    *,
    resolution: float = 1.0,
    seed: int = 1917,
) -> np.ndarray:
    """Partition the graph, returning a community id per node.

    The array is aligned to ``projection.work_ids``. Ids are assigned in
    descending community size (0 is the largest) so numbering is stable across
    runs rather than dependent on iteration order. ``seed`` fixes the
    randomised optimisation.
    """
    adjacency = sp.csr_matrix(projection.adjacency)
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return np.zeros(0, dtype=np.int64)

    # Upper triangle only: the projection is symmetric, and feeding both
    # directions to Leiden would double every edge's weight.
    upper = sp.triu(adjacency, k=1).tocoo()
    sources = upper.row.astype(np.int64)
    targets = upper.col.astype(np.int64)
    weights = upper.data.astype(float)

    membership = _leiden(n_nodes, sources, targets, weights, resolution, seed)
    if membership is None:
        membership = _greedy_modularity(n_nodes, sources, targets, weights, resolution)

    return _renumber_by_size(membership)


def _leiden(
    n_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    resolution: float,
    seed: int,
) -> np.ndarray | None:
    """Leiden partition, or ``None`` if igraph/leidenalg are unavailable."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:  # pragma: no cover - both are hard dependencies
        return None

    graph = ig.Graph(n=n_nodes, edges=list(zip(sources.tolist(), targets.tolist())))
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights.tolist() or None,
        resolution_parameter=resolution,
        # -1 iterates until no further improvement, which is what makes the
        # result reproducible rather than dependent on an iteration budget.
        n_iterations=-1,
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=np.int64)


def _greedy_modularity(
    n_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """NetworkX greedy modularity fallback; fine at test scale, O(n^2) beyond."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for u, v, w in zip(sources.tolist(), targets.tolist(), weights.tolist()):
        graph.add_edge(u, v, weight=w)

    membership = np.zeros(n_nodes, dtype=np.int64)
    communities = nx.community.greedy_modularity_communities(
        graph, weight="weight", resolution=resolution
    )
    for community_id, members in enumerate(communities):
        membership[list(members)] = community_id
    return membership


def _renumber_by_size(membership: np.ndarray) -> np.ndarray:
    """Relabel so 0 is the largest community, ties broken by first appearance.

    Both the map's colour scale and the stored ``communities`` table key off
    these ids, so they must not depend on the optimiser's internal ordering.
    """
    membership = np.asarray(membership, dtype=np.int64)
    values, first_seen, counts = np.unique(
        membership, return_index=True, return_counts=True
    )
    order = np.lexsort((first_seen, -counts))

    remap = np.empty(values.shape[0], dtype=np.int64)
    remap[order] = np.arange(values.shape[0], dtype=np.int64)
    # searchsorted maps each node's old (sorted) label onto its position in
    # ``values``, so this is a single vectorised gather.
    return remap[np.searchsorted(values, membership)]


def modularity(projection: GraphProjection, labels: np.ndarray) -> float:
    """Newman modularity of a partition; used to assert detection quality."""
    adjacency = sp.csr_matrix(projection.adjacency)
    if adjacency.shape[0] == 0:
        return 0.0

    labels = np.asarray(labels)
    # Arbitrary label values (a random partition, say) are compacted to
    # 0..k-1 so they can index the bincount accumulators.
    _, compact = np.unique(labels, return_inverse=True)
    compact = compact.astype(np.int64).ravel()
    n_communities = int(compact.max()) + 1 if compact.size else 0

    strength = np.asarray(adjacency.sum(axis=1)).ravel()
    two_m = float(strength.sum())
    if two_m <= 0:
        return 0.0

    coo = adjacency.tocoo()
    internal = compact[coo.row] == compact[coo.col]
    within = np.bincount(
        compact[coo.row][internal], weights=coo.data[internal], minlength=n_communities
    )
    degree_sum = np.bincount(compact, weights=strength, minlength=n_communities)

    return float(np.sum(within / two_m - (degree_sum / two_m) ** 2))


def label_communities(
    labels: np.ndarray,
    titles: list[str],
    subjects: list[tuple[str, ...]] | None = None,
    *,
    top_k: int = 3,
) -> dict[int, str]:
    """Name each community from the distinctive vocabulary of its members.

    TF-IDF over member titles and subjects, treating each community as one
    document, so a cluster is labelled by what makes it *different* from the
    rest of the graph rather than by generic words like "book" or "novel".
    """
    labels = np.asarray(labels)
    if labels.size == 0 or not titles:
        return {}

    documents: dict[int, Counter[str]] = {}
    for position, label in enumerate(labels.tolist()):
        community = int(label)
        counts = documents.setdefault(community, Counter())
        if position < len(titles):
            counts.update(_tokenize(titles[position]))
        if subjects is not None and position < len(subjects):
            for subject in subjects[position]:
                counts.update(_tokenize(subject))

    n_documents = len(documents)
    document_frequency: Counter[str] = Counter()
    for counts in documents.values():
        document_frequency.update(counts.keys())

    result: dict[int, str] = {}
    for community, counts in sorted(documents.items()):
        # log(N/df) is zero for a term present in every community, which is the
        # whole point: "novel" cannot win a label if every cluster is novels.
        distinctive = [
            (term, frequency * math.log(n_documents / document_frequency[term]))
            for term, frequency in counts.items()
            if document_frequency[term] < n_documents
        ]
        if not distinctive:
            # Single community, or a vocabulary shared by all of them: there is
            # nothing distinctive to say, so fall back to sheer frequency.
            distinctive = [(term, float(frequency)) for term, frequency in counts.items()]

        # Alphabetical tie-break keeps labels stable when scores collide, which
        # they routinely do on small clusters.
        distinctive.sort(key=lambda item: (-item[1], item[0]))
        # Every community gets a name even if its titles reduced to nothing but
        # stopwords, so the map never renders a blank legend entry.
        result[community] = (
            " / ".join(term for term, _ in distinctive[:top_k])
            or f"community {community}"
        )

    return result


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus stopwords and single characters."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


def community_sizes(labels: np.ndarray) -> dict[int, int]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return {}
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}
