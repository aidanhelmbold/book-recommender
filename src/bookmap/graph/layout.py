"""Force-directed layout for the map view.

Layout runs on the *induced subgraph* around the seeds and results -- a few
hundred nodes -- never on the full graph. A 2.4M-node force layout is neither
computable at request time nor legible on screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

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

    ForceAtlas2's force model: repulsion between every pair scaled by degree
    (hubs push their neighbourhoods apart instead of piling on top of them),
    attraction along each edge proportional to its fused weight so strongly
    connected books sit closer together, and a weak gravity that keeps
    disconnected components from drifting off the canvas.

    Repulsion is computed pairwise rather than Barnes-Hut approximated: the
    subgraph is bounded by ``config.max_nodes`` (a few hundred), where one
    vectorised NumPy pass beats a Python-level quadtree outright.
    """
    config = config or LayoutConfig()
    adjacency = sp.csr_matrix(projection.adjacency)
    n_nodes = adjacency.shape[0]

    if n_nodes == 0:
        return np.zeros((0, 2), dtype=float)
    if n_nodes == 1:
        return np.zeros((1, 2), dtype=float)

    rng = np.random.default_rng(config.seed)
    # Seeded on a circle plus jitter rather than pure noise: a spread-out start
    # avoids the huge initial repulsion spikes that come from coincident nodes.
    angles = rng.permutation(n_nodes) * (2.0 * np.pi / n_nodes)
    radius = np.sqrt(n_nodes)
    positions = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles))
    ) + rng.normal(scale=0.1, size=(n_nodes, 2))

    coo = adjacency.tocoo()
    edge_rows = coo.row.astype(np.intp)
    edge_cols = coo.col.astype(np.intp)
    edge_weights = coo.data.astype(float)

    # FA2 "mass": degree + 1, so a hub both repels harder and resists being
    # dragged into a leaf's orbit.
    mass = np.asarray((adjacency != 0).sum(axis=1), dtype=float).ravel() + 1.0
    mass_product = np.outer(mass, mass)

    repulsion_strength = float(config.scaling_ratio)
    gravity = float(config.gravity)
    iterations = max(int(config.iterations), 1)

    for step in range(iterations):
        delta = positions[:, None, :] - positions[None, :, :]
        distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
        # Self-pairs contribute nothing; the floor keeps near-coincident nodes
        # from producing an infinite force (the classic source of NaN layouts).
        np.fill_diagonal(distance_squared, np.inf)
        np.maximum(distance_squared, 1e-9, out=distance_squared)

        # |F| = k * m_i * m_j / d, directed along delta/d, hence delta / d^2.
        repulsion = np.einsum(
            "ij,ijk->ik", repulsion_strength * mass_product / distance_squared, delta
        )

        # Linear attraction: |F| = w * d, pulling each endpoint toward the other.
        # The adjacency holds both directions, so one pass covers both endpoints.
        edge_delta = positions[edge_rows] - positions[edge_cols]
        attraction = np.zeros_like(positions)
        np.add.at(attraction, edge_rows, -edge_weights[:, None] * edge_delta)

        centre_distance = np.linalg.norm(positions, axis=1, keepdims=True)
        pull = -gravity * mass[:, None] * positions / np.maximum(centre_distance, 1e-9)

        force = repulsion + attraction + pull
        magnitude = np.linalg.norm(force, axis=1, keepdims=True)
        # Bounded, cooling steps: dividing by (1 + |F|) caps a single node's
        # movement however large its force, which is what keeps the simulation
        # finite without hand-tuned per-graph step sizes.
        cooling = 1.0 - step / iterations
        positions += force * (cooling / (1.0 + magnitude))

    # Centre the drawing so downstream scaling is about spread, not offset.
    return positions - positions.mean(axis=0)


def normalize_positions(
    positions: np.ndarray,
    *,
    width: float = 1000.0,
    height: float = 1000.0,
    padding: float = 40.0,
) -> np.ndarray:
    """Scale positions into a fixed viewport, preserving aspect ratio."""
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.zeros((0, 2), dtype=float)

    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    span = upper - lower

    available = np.array(
        [max(width - 2.0 * padding, 0.0), max(height - 2.0 * padding, 0.0)]
    )
    # One scale factor for both axes, chosen by the tighter constraint: scaling
    # x and y independently would stretch the graph and destroy the relative
    # distances that are the whole payload of a map.
    candidates = [available[axis] / span[axis] for axis in (0, 1) if span[axis] > 0]
    scale = min(candidates) if candidates else 1.0

    centre = (lower + upper) / 2.0
    viewport_centre = np.array([width / 2.0, height / 2.0])
    return (positions - centre) * scale + viewport_centre
