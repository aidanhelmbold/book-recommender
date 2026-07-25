"""The recommendation pipeline: a seed set in, a ranked explained list out.

    resolve seeds -> PPR over the whole set -> hub damping -> filter -> MMR -> explain

Each stage exists to fix a specific failure of the previous one. PPR alone
returns bestsellers; damping fixes that but leaves results clustered in one
genre; MMR spreads them across the regions the seeds actually touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bookmap.config import RecommendConfig
from bookmap.models import Recommendation, SeedMatch

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.db import Store
    from bookmap.store.projection import GraphProjection


@dataclass(frozen=True, slots=True)
class RecommendResult:
    """Recommendations plus enough context to explain and draw them."""

    recommendations: tuple[Recommendation, ...]
    seeds: tuple[SeedMatch, ...]
    unresolved: tuple[str, ...] = ()
    """Queries that matched nothing -- surfaced, never silently dropped."""

    ambiguous: tuple[SeedMatch, ...] = ()
    """Queries with near-tied matches, for the caller to disambiguate."""


def resolve_seeds(
    queries: list[str],
    store: Store,
    *,
    threshold: float = 0.75,
) -> tuple[list[SeedMatch], list[str]]:
    """Resolve typed titles onto works. Returns ``(matches, unresolved)``."""
    raise NotImplementedError


def mmr_rerank(
    candidate_rows: np.ndarray,
    scores: np.ndarray,
    projection: GraphProjection,
    *,
    n: int = 20,
    lambda_: float = 0.7,
) -> list[int]:
    """Greedy maximal-marginal-relevance selection over graph neighbourhoods.

    ``candidate_rows`` holds the projection rows eligible for selection.
    ``scores`` is a *full-graph* array aligned to ``projection.work_ids`` (the
    shape :func:`bookmap.graph.ppr.hub_damped_scores` returns), so no re-indexing
    is needed between stages. Returns the chosen rows in selection order.

    Selects the row maximising::

        lambda_ * score(v) - (1 - lambda_) * max_{u in selected} jaccard(N(v), N(u))

    Similarity is Jaccard overlap of adjacency, so two books with nearly the
    same neighbours are treated as redundant even if never directly linked --
    which is what stops a result set being five editions of one idea.
    """
    raise NotImplementedError


def recommend(
    seed_queries: list[str],
    store: Store,
    projection: GraphProjection,
    *,
    n: int = 20,
    config: RecommendConfig | None = None,
) -> RecommendResult:
    """Run the full pipeline.

    Seeds themselves are always excluded from the output -- recommending a book
    the user already listed is the most obvious possible failure.
    """
    raise NotImplementedError
