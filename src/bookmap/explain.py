"""Explain a recommendation by the path that produced it.

"Because it is two hops from *Hyperion* and adjacent to *Blindsight*" is both
more convincing and more debuggable than a bare score. Explanations are also how
you notice the graph is wrong: a nonsense path means a bad edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bookmap.models import Explanation

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.db import Store
    from bookmap.store.projection import GraphProjection


def edge_cost(weight: float) -> float:
    """Convert an edge weight to a shortest-path cost via ``-log(weight)``.

    Under this transform the lowest-cost path is the *highest-probability* chain
    of connections, since adding logs multiplies probabilities -- so a two-hop
    path through strong links can legitimately beat a weak direct link.
    """
    raise NotImplementedError


def explain_paths(
    target: str,
    seeds: list[str],
    projection: GraphProjection,
    store: Store,
    *,
    max_paths: int = 3,
    max_hops: int = 4,
) -> tuple[Explanation, ...]:
    """Find the strongest paths from distinct seeds to a recommendation.

    Returns at most ``max_paths`` explanations, each from a *different* seed
    (three paths from the same seed say little), strongest first. Paths longer
    than ``max_hops`` are omitted as unconvincing rather than padded.
    """
    raise NotImplementedError


def format_explanation(explanation: Explanation, titles: dict[str, str]) -> str:
    """Render an explanation as a human-readable line for CLI output."""
    raise NotImplementedError
