"""Degree, betweenness, and bridge-book detection.

The two-cluster fixture is built so the answers are obvious by inspection: every
path between the science-fiction clique and the literary clique runs through one
node, so that node must dominate betweenness and top the bridge ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from bookmap.graph.centrality import (
    approximate_betweenness,
    bridge_books,
    degrees,
    weighted_degrees,
)
from bookmap.graph.communities import detect_communities
from tests.helpers import make_projection


class TestDegrees:
    def test_unweighted(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        degree = degrees(projection)
        # sf1 is in a 4-clique (3 neighbours) plus the bridge.
        assert degree[projection.index["sf1"]] == 4
        assert degree[projection.index["bridge"]] == 2

    def test_weighted(self) -> None:
        projection = make_projection([("a", "b", 2.0), ("a", "c", 0.5)])
        assert weighted_degrees(projection)[projection.index["a"]] == pytest.approx(2.5)

    def test_aligned_to_work_ids(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        assert degrees(projection).shape == (projection.n_nodes,)


class TestBetweenness:
    def test_bridge_dominates(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        scores = approximate_betweenness(projection, k_samples=projection.n_nodes)
        bridge = scores[projection.index["bridge"]]
        others = [
            scores[projection.index[n]]
            for n in projection.work_ids
            if n != "bridge"
        ]
        assert bridge > max(others)

    def test_matches_networkx_when_exact(self, two_cluster_pairs) -> None:
        """With k_samples >= n_nodes the result must be exact betweenness."""
        import networkx as nx

        projection = make_projection(two_cluster_pairs)
        got = approximate_betweenness(projection, k_samples=projection.n_nodes)
        expected = nx.betweenness_centrality(projection.to_networkx(), normalized=True)
        for work_id, row in projection.index.items():
            assert got[row] == pytest.approx(expected[work_id], abs=1e-6)

    def test_sampling_ranks_the_bridge_first(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        scores = approximate_betweenness(projection, k_samples=5, seed=3)
        assert int(np.argmax(scores)) == projection.index["bridge"]

    def test_deterministic_for_fixed_seed(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        first = approximate_betweenness(projection, k_samples=5, seed=11)
        second = approximate_betweenness(projection, k_samples=5, seed=11)
        assert np.allclose(first, second)


class TestBridgeBooks:
    def test_finds_the_bridge(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        labels = detect_communities(projection)
        bridges = bridge_books(projection, labels, top_n=3)
        assert bridges
        assert bridges[0][0] == "bridge"

    def test_reports_communities_touched(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        labels = detect_communities(projection)
        work_id, score, touched = bridge_books(projection, labels, top_n=1)[0]
        assert work_id == "bridge"
        assert score > 0
        assert len(set(touched)) >= 2

    def test_respects_top_n(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        labels = detect_communities(projection)
        assert len(bridge_books(projection, labels, top_n=2)) <= 2

    def test_interior_nodes_score_lower_than_the_bridge(self, two_cluster_pairs) -> None:
        projection = make_projection(two_cluster_pairs)
        labels = detect_communities(projection)
        ranked = bridge_books(projection, labels, top_n=99)
        scores = dict((w, s) for w, s, _ in ranked)
        assert scores["bridge"] > scores.get("sf4", 0.0)

    def test_single_community_has_no_bridges(self) -> None:
        projection = make_projection([("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 1.0)])
        labels = np.zeros(projection.n_nodes, dtype=int)
        assert bridge_books(projection, labels) == []
