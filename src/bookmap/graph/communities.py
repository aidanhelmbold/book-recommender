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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.projection import GraphProjection


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
    raise NotImplementedError


def modularity(projection: GraphProjection, labels: np.ndarray) -> float:
    """Newman modularity of a partition; used to assert detection quality."""
    raise NotImplementedError


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
    raise NotImplementedError


def community_sizes(labels: np.ndarray) -> dict[int, int]:
    raise NotImplementedError
