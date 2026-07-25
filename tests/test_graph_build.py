"""Fusion arithmetic, checked against values computed by hand.

Every expected number below was worked out independently of the implementation.
That is deliberate: a test that recomputes the formula it is checking proves
nothing.
"""

from __future__ import annotations

import pytest

from bookmap.config import GraphConfig
from bookmap.graph.build import fuse_edges, prune, rank_weight
from bookmap.models import Edge, EdgeKind, FusedEdge, SourceName

GR = EdgeKind.GR_SIMILAR
AZB = EdgeKind.AZ_ALSO_BOUGHT
AZV = EdgeKind.AZ_ALSO_VIEWED

# 1 / log2(2 + rank), hand-computed at the ranks with exact values.
RANK_1 = 0.6309297535714574   # 1 / log2(3)
RANK_2 = 0.5                  # 1 / log2(4)
RANK_6 = 1.0 / 3.0            # 1 / log2(8)
RANK_14 = 0.25                # 1 / log2(16)
RANK_30 = 0.2                 # 1 / log2(32)


class TestRankWeight:
    @pytest.mark.parametrize(
        ("rank", "expected"),
        [(1, RANK_1), (2, RANK_2), (6, RANK_6), (14, RANK_14), (30, RANK_30)],
    )
    def test_known_values(self, rank: int, expected: float) -> None:
        assert rank_weight(rank) == pytest.approx(expected, rel=1e-12)

    def test_strictly_decreasing(self) -> None:
        weights = [rank_weight(r) for r in range(1, 60)]
        assert all(a > b for a, b in zip(weights, weights[1:], strict=False))

    def test_decay_is_gentle_through_the_middle(self) -> None:
        """Position 10 should be worth roughly half position 1, not a tenth.

        This is why log decay was chosen over 1/rank; if someone swaps in a
        harsher curve the middle of every "also bought" list stops counting.
        """
        # 1/log2(12) / (1/log2(3)) = 0.2789429456511298 / 0.6309297535714574
        assert rank_weight(10) / rank_weight(1) == pytest.approx(0.4421, abs=1e-3)

    def test_rejects_zero_and_negative(self) -> None:
        for bad in (0, -1):
            with pytest.raises(ValueError):
                rank_weight(bad)


class TestFuseEdges:
    def test_single_directed_edge_is_fully_asymmetric(self) -> None:
        # 0.6 (gr alpha) * 0.6309297535714574 (rank 1) = 0.37855785214287444
        fused = list(fuse_edges([Edge("a", "b", GR, 1, SourceName.GOODREADS_UCSD)]))
        assert len(fused) == 1
        edge = fused[0]
        assert edge.weight == pytest.approx(0.37855785214287444, rel=1e-12)
        assert edge.dir_asym == pytest.approx(1.0)

    def test_mutual_pair_takes_max_and_records_asymmetry(self) -> None:
        """a->b at rank 1, b->a at rank 2.

        w(a,b) = 0.6 * 0.6309297535714574 = 0.37855785214287444
        w(b,a) = 0.6 * 0.5               = 0.3
        weight   = max                    = 0.37855785214287444
        dir_asym = 1 - 0.3/0.37855785...  = 0.2075187496394219
        """
        fused = list(
            fuse_edges(
                [
                    Edge("a", "b", GR, 1, SourceName.GOODREADS_UCSD),
                    Edge("b", "a", GR, 2, SourceName.GOODREADS_UCSD),
                ]
            )
        )
        assert len(fused) == 1
        assert fused[0].weight == pytest.approx(0.37855785214287444, rel=1e-12)
        assert fused[0].dir_asym == pytest.approx(0.2075187496394219, rel=1e-9)

    def test_perfectly_mutual_pair_has_zero_asymmetry(self) -> None:
        fused = list(
            fuse_edges(
                [
                    Edge("a", "b", GR, 3, SourceName.GOODREADS_UCSD),
                    Edge("b", "a", GR, 3, SourceName.GOODREADS_UCSD),
                ]
            )
        )
        assert fused[0].dir_asym == pytest.approx(0.0)

    def test_sources_add_within_a_direction(self) -> None:
        """An edge attested by Goodreads *and* Amazon must outrank either alone.

        w(a,b) = 0.6 * 0.6309297535714574 + 0.3 * 0.6309297535714574
               = 0.9 * 0.6309297535714574 = 0.5678367782143117
        """
        fused = list(
            fuse_edges(
                [
                    Edge("a", "b", GR, 1, SourceName.GOODREADS_UCSD),
                    Edge("a", "b", AZB, 1, SourceName.AMAZON_META),
                ]
            )
        )
        assert len(fused) == 1
        assert fused[0].weight == pytest.approx(0.5678367782143117, rel=1e-12)
        assert set(fused[0].kinds) == {GR, AZB}

    def test_also_viewed_is_much_weaker_than_also_bought(self) -> None:
        bought = list(fuse_edges([Edge("a", "b", AZB, 1, SourceName.AMAZON_META)]))[0]
        viewed = list(fuse_edges([Edge("a", "b", AZV, 1, SourceName.AMAZON_META)]))[0]
        assert bought.weight == pytest.approx(3.0 * viewed.weight, rel=1e-9)

    def test_output_is_canonically_ordered(self) -> None:
        """src < dst always, so an undirected pair can never appear twice."""
        fused = list(
            fuse_edges(
                [
                    Edge("zeta", "alpha", GR, 1, SourceName.GOODREADS_UCSD),
                    Edge("alpha", "zeta", GR, 1, SourceName.GOODREADS_UCSD),
                ]
            )
        )
        assert len(fused) == 1
        assert fused[0].src == "alpha"
        assert fused[0].dst == "zeta"

    def test_min_weight_prunes(self) -> None:
        # OL_SUBJECT alpha is 0.05; at rank 30 that is 0.05*0.2 = 0.01 < 0.05.
        edges = [Edge("a", "b", EdgeKind.OL_SUBJECT, 30, SourceName.OPENLIBRARY)]
        assert list(fuse_edges(edges)) == []
        loose = GraphConfig(min_weight=0.001)
        assert len(list(fuse_edges(edges, loose))) == 1

    def test_max_rank_ignores_the_tail(self) -> None:
        config = GraphConfig(max_rank=5, min_weight=0.0)
        edges = [
            Edge("a", "b", GR, 3, SourceName.GOODREADS_UCSD),
            Edge("a", "c", GR, 40, SourceName.GOODREADS_UCSD),
        ]
        fused = list(fuse_edges(edges, config))
        assert {(e.src, e.dst) for e in fused} == {("a", "b")}

    def test_empty_input(self) -> None:
        assert list(fuse_edges([])) == []


class TestPrune:
    def test_drops_weak_edges(self) -> None:
        edges = [
            FusedEdge("a", "b", 0.4),
            FusedEdge("b", "c", 0.01),
        ]
        kept = prune(edges, min_weight=0.05)
        assert [(e.src, e.dst) for e in kept] == [("a", "b")]

    def test_drop_leaves_removes_degree_one_nodes(self) -> None:
        """b and c form a triangle with d; 'leaf' hangs off b alone."""
        edges = [
            FusedEdge("b", "c", 0.5),
            FusedEdge("c", "d", 0.5),
            FusedEdge("b", "d", 0.5),
            FusedEdge("b", "leaf", 0.5),
        ]
        kept = prune(edges, min_weight=0.0, drop_leaves=True)
        remaining = {e.src for e in kept} | {e.dst for e in kept}
        assert "leaf" not in remaining
        assert remaining == {"b", "c", "d"}

    def test_leaves_kept_by_default(self) -> None:
        edges = [FusedEdge("b", "c", 0.5), FusedEdge("b", "leaf", 0.5)]
        kept = prune(edges, min_weight=0.0)
        assert len(kept) == 2
