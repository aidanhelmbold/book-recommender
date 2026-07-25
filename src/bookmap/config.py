"""Tunable parameters for graph fusion and recommendation.

Every number that shapes a recommendation lives here rather than being buried
in an algorithm, so the ranking can be retuned without touching logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bookmap.models import EdgeKind

SOURCE_ALPHA: dict[EdgeKind, float] = {
    EdgeKind.GR_SIMILAR: 0.6,
    EdgeKind.AZ_ALSO_BOUGHT: 0.3,
    EdgeKind.AZ_ALSO_VIEWED: 0.1,
    EdgeKind.OL_SUBJECT: 0.05,
}
"""Per-source trust. Goodreads' similar-books list is a curated recommender
output; Amazon co-purchase is noisier (bundles, gifts, textbooks); shared
subjects are a topical fallback, not evidence of reader behaviour."""


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Controls how raw directed edges become the fused undirected graph."""

    source_alpha: dict[EdgeKind, float] = field(
        default_factory=lambda: dict(SOURCE_ALPHA)
    )
    min_weight: float = 0.05
    """Fused edges below this are pruned as noise."""

    drop_leaves: bool = False
    """Drop degree-1 nodes; tightens the map but loses long-tail books."""

    max_rank: int = 50
    """Ignore list positions beyond this — the tail of an "also bought" list is
    mostly unrelated inventory."""


@dataclass(frozen=True, slots=True)
class RecommendConfig:
    """Controls the ranking pipeline."""

    ppr_alpha: float = 0.15
    """Restart probability. Lower explores further from the seeds."""

    ppr_tol: float = 1e-8
    ppr_max_iter: int = 200

    hub_beta: float = 0.25
    """Hub damping exponent in ``score = ppr / degree**beta``. This is the main
    quality lever: at 0 the results are dominated by bestsellers for every
    conceivable seed set."""

    mmr_lambda: float = 0.7
    """Relevance/diversity tradeoff in MMR reranking. 1.0 disables diversity."""

    exclude_same_author: bool = False
    """Filter other books by a seed's author — useful when you want to discover
    new writers rather than the rest of a series."""

    max_explanations: int = 3
    """Distinct seeds to trace a path back from, per recommendation."""


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    """Controls the force-directed layout used by the map."""

    iterations: int = 200
    seed: int = 1917
    """Fixed so a given subgraph always lays out identically."""

    scaling_ratio: float = 2.0
    gravity: float = 1.0
    max_nodes: int = 800
    """Above this the induced subgraph is trimmed by score before layout."""


DEFAULT_DB_PATH = "bookmap.duckdb"
