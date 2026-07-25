"""Property-based tests for invariants that examples cover badly.

The ingest-idempotency property is the important one. Fusion sums over
``edges_raw``, so any path that lets a re-ingest duplicate rows inflates edge
weights silently -- no exception, no obviously wrong output, just a graph that
drifts every time someone reloads a dump. Hypothesis is a better tool for that
than a handful of chosen cases.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bookmap.graph.build import fuse_edges, rank_weight
from bookmap.graph.ppr import personalized_pagerank
from bookmap.models import Book, Edge, EdgeKind, SourceName
from bookmap.store.db import Store
from tests.helpers import csr_from_pairs

node_names = st.text(alphabet="abcdefgh", min_size=1, max_size=2)
ranks = st.integers(min_value=1, max_value=50)
kinds = st.sampled_from(list(EdgeKind))


@st.composite
def edge_lists(draw, min_size: int = 1, max_size: int = 20) -> list[Edge]:
    raw = draw(
        st.lists(
            st.tuples(node_names, node_names, kinds, ranks),
            min_size=min_size,
            max_size=max_size,
        )
    )
    edges = [
        Edge(src, dst, kind, rank, SourceName.DEMO)
        for src, dst, kind, rank in raw
        if src != dst
    ]
    assume(edges)
    return edges


class TestRankWeightProperties:
    @given(ranks, ranks)
    def test_monotonically_decreasing(self, a: int, b: int) -> None:
        if a < b:
            assert rank_weight(a) > rank_weight(b)
        elif a > b:
            assert rank_weight(a) < rank_weight(b)
        else:
            assert rank_weight(a) == rank_weight(b)

    @given(ranks)
    def test_bounded_between_zero_and_one(self, rank: int) -> None:
        assert 0.0 < rank_weight(rank) <= 1.0


class TestFusionProperties:
    @given(edge_lists())
    @settings(max_examples=50, deadline=None)
    def test_output_is_canonically_ordered(self, edges: list[Edge]) -> None:
        for fused in fuse_edges(edges):
            assert fused.src < fused.dst

    @given(edge_lists())
    @settings(max_examples=50, deadline=None)
    def test_no_duplicate_pairs(self, edges: list[Edge]) -> None:
        fused = list(fuse_edges(edges))
        pairs = [(e.src, e.dst) for e in fused]
        assert len(pairs) == len(set(pairs))

    @given(edge_lists())
    @settings(max_examples=50, deadline=None)
    def test_weights_are_positive_and_asymmetry_is_bounded(self, edges: list[Edge]) -> None:
        for fused in fuse_edges(edges):
            assert fused.weight > 0
            assert 0.0 <= fused.dir_asym <= 1.0

    @given(edge_lists())
    @settings(max_examples=50, deadline=None)
    def test_fusion_is_idempotent(self, edges: list[Edge]) -> None:
        """Fusing the fused input's source twice must not change anything.

        Guards against an accumulator that keeps state between calls.
        """
        first = {(e.src, e.dst): e.weight for e in fuse_edges(edges)}
        second = {(e.src, e.dst): e.weight for e in fuse_edges(edges)}
        assert first == second

    @given(edge_lists())
    @settings(max_examples=50, deadline=None)
    def test_input_order_does_not_matter(self, edges: list[Edge]) -> None:
        forward = {(e.src, e.dst): round(e.weight, 12) for e in fuse_edges(edges)}
        backward = {
            (e.src, e.dst): round(e.weight, 12) for e in fuse_edges(list(reversed(edges)))
        }
        assert forward == backward


class TestPPRProperties:
    @given(
        st.lists(
            st.tuples(node_names, node_names),
            min_size=1,
            max_size=15,
        ),
        st.floats(min_value=0.05, max_value=0.5),
    )
    @settings(max_examples=50, deadline=None)
    def test_is_always_a_probability_distribution(
        self, raw_pairs: list[tuple[str, str]], alpha: float
    ) -> None:
        pairs = [(u, v, 1.0) for u, v in raw_pairs if u != v]
        assume(pairs)
        adjacency, work_ids, index = csr_from_pairs(pairs)
        ppr = personalized_pagerank(adjacency, [0], alpha=alpha)
        assert (ppr >= -1e-12).all()
        assert abs(ppr.sum() - 1.0) < 1e-6

    @given(st.floats(min_value=0.05, max_value=0.5))
    @settings(max_examples=20, deadline=None)
    def test_seed_retains_the_most_mass_in_a_star(self, alpha: float) -> None:
        pairs = [("hub", f"leaf{i}", 1.0) for i in range(5)]
        adjacency, work_ids, index = csr_from_pairs(pairs)
        ppr = personalized_pagerank(adjacency, [index["leaf0"]], alpha=alpha)
        assert ppr[index["leaf0"]] == max(ppr)


class TestIngestIdempotencyProperty:
    @given(edges=edge_lists(min_size=1, max_size=12))
    @settings(max_examples=25, deadline=None)
    def test_reingesting_never_changes_the_graph(self, tmp_path_factory, edges: list[Edge]) -> None:
        """The bug this guards is silent: duplicated rows inflate weights.

        Ingest once, snapshot the fused weights, ingest the identical data again,
        and the weights must be byte-identical -- not merely close.
        """
        path = tmp_path_factory.mktemp("idem") / "t.duckdb"
        names = {e.src for e in edges} | {e.dst for e in edges}
        books = [Book(work_id=n, title=f"Book {n}") for n in names]

        with Store.open(path) as store:
            store.upsert_books(books)
            store.insert_raw_edges(edges)
            store.replace_fused_edges(fuse_edges(store.iter_raw_edges()))
            first = {(e.src, e.dst): e.weight for e in store.iter_fused_edges()}

            store.upsert_books(books)
            store.insert_raw_edges(edges)
            store.replace_fused_edges(fuse_edges(store.iter_raw_edges()))
            second = {(e.src, e.dst): e.weight for e in store.iter_fused_edges()}

        assert first == second

    @given(edges=edge_lists(min_size=1, max_size=12))
    @settings(max_examples=25, deadline=None)
    def test_row_counts_are_stable_across_reingest(
        self, tmp_path_factory, edges: list[Edge]
    ) -> None:
        path = tmp_path_factory.mktemp("idem2") / "t.duckdb"
        names = {e.src for e in edges} | {e.dst for e in edges}
        books = [Book(work_id=n, title=f"Book {n}") for n in names]
        with Store.open(path) as store:
            store.upsert_books(books)
            store.insert_raw_edges(edges)
            before = store.counts()
            store.upsert_books(books)
            store.insert_raw_edges(edges)
            assert store.counts() == before


class TestHubDampingProperties:
    @given(
        st.lists(st.floats(min_value=1e-6, max_value=1.0), min_size=2, max_size=20),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_never_produces_nan_or_negative(self, ppr_values: list[float], beta: float) -> None:
        from bookmap.graph.ppr import hub_damped_scores

        ppr = np.array(ppr_values)
        degree = np.arange(1, len(ppr_values) + 1, dtype=float)
        scores = hub_damped_scores(ppr, degree, beta=beta)
        assert np.isfinite(scores).all()
        assert (scores >= 0).all()

    @given(st.floats(min_value=0.01, max_value=1.0))
    @settings(max_examples=30, deadline=None)
    def test_damping_is_monotone_in_degree(self, ppr_value: float) -> None:
        """For equal PPR mass, a higher-degree node must never score higher."""
        from bookmap.graph.ppr import hub_damped_scores

        ppr = np.array([ppr_value, ppr_value])
        scores = hub_damped_scores(ppr, np.array([2.0, 200.0]), beta=0.25)
        assert scores[0] >= scores[1]
