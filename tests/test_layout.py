"""Layout is tested by invariant, not by coordinate.

Asserting exact positions from a force simulation would pin an arbitrary
implementation detail and break on any tuning change. What actually matters is
that the layout is finite, bounded, reproducible, and that connected books end
up nearer each other than unconnected ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from bookmap.config import LayoutConfig
from bookmap.graph.layout import force_atlas2, normalize_positions
from tests.helpers import make_projection


class TestForceAtlas2:
    def test_shape(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        positions = force_atlas2(projection)
        assert positions.shape == (projection.n_nodes, 2)

    def test_all_finite(self, two_cluster_pairs) -> None:
        """Force simulations diverge to inf/nan when repulsion is mishandled."""
        projection = make_projection(two_cluster_pairs)
        assert np.isfinite(force_atlas2(projection)).all()

    def test_deterministic_for_fixed_seed(self, two_cluster_pairs) -> None:
        """The same query must always draw the same picture."""
        projection = make_projection(two_cluster_pairs)
        config = LayoutConfig(seed=42, iterations=50)
        assert np.allclose(force_atlas2(projection, config), force_atlas2(projection, config))

    def test_different_seeds_differ(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        a = force_atlas2(projection, LayoutConfig(seed=1, iterations=50))
        b = force_atlas2(projection, LayoutConfig(seed=2, iterations=50))
        assert not np.allclose(a, b)

    def test_nodes_do_not_collapse_to_one_point(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        positions = force_atlas2(projection)
        assert positions.std(axis=0).min() > 1e-6

    def test_clusters_separate(self, two_cluster_pairs) -> None:
        """Within-cluster distances should beat across-cluster distances.

        This is the one thing a layout is for: if it fails, the map conveys
        nothing even though every other invariant holds.
        """
        projection = make_projection(two_cluster_pairs)
        positions = force_atlas2(projection, LayoutConfig(seed=5, iterations=400))
        index = projection.index
        sf = [index[n] for n in ("sf1", "sf2", "sf3", "sf4")]
        lit = [index[n] for n in ("lit1", "lit2", "lit3", "lit4")]

        def mean_distance(rows_a: list[int], rows_b: list[int]) -> float:
            return float(
                np.mean(
                    [
                        np.linalg.norm(positions[i] - positions[j])
                        for i in rows_a
                        for j in rows_b
                        if i != j
                    ]
                )
            )

        within = (mean_distance(sf, sf) + mean_distance(lit, lit)) / 2
        across = mean_distance(sf, lit)
        assert across > within

    def test_single_node(self) -> None:
        projection = make_projection([], nodes=["only"])
        assert force_atlas2(projection).shape == (1, 2)

    def test_empty_graph(self) -> None:
        projection = make_projection([], nodes=[])
        assert force_atlas2(projection).shape == (0, 2)

    def test_disconnected_nodes_are_still_placed(self) -> None:
        projection = make_projection([("a", "b", 1.0)], nodes=["a", "b", "island"])
        positions = force_atlas2(projection)
        assert np.isfinite(positions).all()


class TestNormalizePositions:
    def test_fits_inside_the_viewport(self) -> None:
        positions = np.array([[-500.0, 12.0], [900.0, -3.0], [0.0, 400.0]])
        scaled = normalize_positions(positions, width=1000, height=1000, padding=40)
        assert scaled[:, 0].min() >= 40 - 1e-9
        assert scaled[:, 0].max() <= 1000 - 40 + 1e-9
        assert scaled[:, 1].min() >= 40 - 1e-9
        assert scaled[:, 1].max() <= 1000 - 40 + 1e-9

    def test_preserves_aspect_ratio(self) -> None:
        """A wide graph must not be stretched into a square; relative distances
        are the information the map carries."""
        positions = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 10.0]])
        scaled = normalize_positions(positions, width=1000, height=1000, padding=0)
        wide = np.linalg.norm(scaled[1] - scaled[0])
        tall = np.linalg.norm(scaled[2] - scaled[0])
        assert wide / tall == pytest.approx(10.0, rel=1e-6)

    def test_handles_identical_points(self) -> None:
        positions = np.array([[5.0, 5.0], [5.0, 5.0]])
        scaled = normalize_positions(positions)
        assert np.isfinite(scaled).all()

    def test_single_point_is_centred(self) -> None:
        scaled = normalize_positions(np.array([[3.0, 7.0]]), width=1000, height=600)
        assert scaled[0][0] == pytest.approx(500.0)
        assert scaled[0][1] == pytest.approx(300.0)

    def test_empty(self) -> None:
        assert normalize_positions(np.zeros((0, 2))).shape == (0, 2)
