"""Force-directed layout for the map view.

Layout runs on the *induced subgraph* around the seeds and results -- a few
hundred nodes -- never on the full graph. A 2.4M-node force layout is neither
computable at request time nor legible on screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bookmap.config import LayoutConfig

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.projection import GraphProjection


def force_atlas2(
    projection: GraphProjection,
    config: LayoutConfig | None = None,
) -> np.ndarray:
    """Compute 2D positions, returning an ``(n_nodes, 2)`` float array.

    Deterministic for a fixed ``config.seed``: the same subgraph must always
    produce the same picture, or the map appears to reshuffle itself whenever a
    user re-runs a query.

    Repulsion is Barnes-Hut approximated; attraction follows edge weight, so
    strongly connected books sit closer together.
    """
    raise NotImplementedError


def normalize_positions(
    positions: np.ndarray,
    *,
    width: float = 1000.0,
    height: float = 1000.0,
    padding: float = 40.0,
) -> np.ndarray:
    """Scale positions into a fixed viewport, preserving aspect ratio."""
    raise NotImplementedError
