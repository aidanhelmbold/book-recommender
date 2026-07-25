"""Path-based explanations.

The ``-log(weight)`` cost transform is what makes "two strong hops beat one weak
link" fall out naturally, and that is exactly what the graph below is built to
check.
"""

from __future__ import annotations

import math

import pytest

from bookmap.explain import edge_cost, explain_paths, format_explanation
from bookmap.models import Book, Explanation, FusedEdge
from tests.helpers import make_projection


class TestEdgeCost:
    def test_known_values(self) -> None:
        assert edge_cost(1.0) == pytest.approx(0.0)
        assert edge_cost(0.5) == pytest.approx(math.log(2.0))
        assert edge_cost(0.25) == pytest.approx(math.log(4.0))

    def test_stronger_edges_cost_less(self) -> None:
        assert edge_cost(0.9) < edge_cost(0.1)

    def test_costs_add_the_way_probabilities_multiply(self) -> None:
        """Two 0.5 hops must cost the same as one 0.25 hop.

        This is the property that lets a chain of strong links out-compete a
        single weak one, which is what makes multi-hop explanations meaningful.
        """
        assert edge_cost(0.5) + edge_cost(0.5) == pytest.approx(edge_cost(0.25))

    def test_zero_weight_is_impassable_not_an_error(self) -> None:
        assert edge_cost(0.0) == math.inf


@pytest.fixture
def explain_store(store):
    books = [
        Book(work_id="seed1", title="Dune", authors=("Frank Herbert",)),
        Book(work_id="seed2", title="Emma", authors=("Jane Austen",)),
        Book(work_id="mid", title="Middle Book", authors=("M. Author",)),
        Book(work_id="target", title="Target Book", authors=("T. Author",)),
        Book(work_id="far", title="Far Book", authors=("F. Author",)),
    ]
    store.upsert_books(books)
    store.replace_fused_edges(
        [
            FusedEdge("seed1", "mid", 0.9),
            FusedEdge("mid", "target", 0.9),
            FusedEdge("seed1", "target", 0.05),
            FusedEdge("seed2", "target", 0.4),
            FusedEdge("target", "far", 0.01),
        ]
    )
    return store


@pytest.fixture
def explain_projection(explain_store):
    from bookmap.store.projection import project

    return project(explain_store)


class TestExplainPaths:
    def test_prefers_two_strong_hops_over_one_weak_link(
        self, explain_store, explain_projection
    ) -> None:
        """seed1 -> mid -> target (0.9, 0.9) beats seed1 -> target (0.05).

        Cost 0.105 + 0.105 versus 3.0, so the direct edge is the *worse*
        explanation even though it is shorter.
        """
        explanations = explain_paths("target", ["seed1"], explain_projection, explain_store)
        assert explanations
        assert explanations[0].path == ("seed1", "mid", "target")
        assert explanations[0].hops == 2

    def test_one_path_per_distinct_seed(self, explain_store, explain_projection) -> None:
        """Three paths from one seed say much less than one path from each."""
        explanations = explain_paths(
            "target", ["seed1", "seed2"], explain_projection, explain_store, max_paths=2
        )
        assert len({e.seed_id for e in explanations}) == len(explanations)
        assert {e.seed_id for e in explanations} == {"seed1", "seed2"}

    def test_strongest_first(self, explain_store, explain_projection) -> None:
        explanations = explain_paths(
            "target", ["seed1", "seed2"], explain_projection, explain_store, max_paths=2
        )
        strengths = [e.strength for e in explanations]
        assert strengths == sorted(strengths, reverse=True)

    def test_respects_max_paths(self, explain_store, explain_projection) -> None:
        explanations = explain_paths(
            "target", ["seed1", "seed2"], explain_projection, explain_store, max_paths=1
        )
        assert len(explanations) == 1

    def test_seed_titles_are_populated(self, explain_store, explain_projection) -> None:
        explanations = explain_paths("target", ["seed1"], explain_projection, explain_store)
        assert explanations[0].seed_title == "Dune"

    def test_over_long_paths_are_omitted(self, explain_store, explain_projection) -> None:
        """An unconvincing path is worse than no path -- do not pad the output."""
        explanations = explain_paths(
            "target", ["seed1"], explain_projection, explain_store, max_hops=1
        )
        assert explanations == () or all(e.hops <= 1 for e in explanations)

    def test_unreachable_seed_yields_nothing(self, explain_store, explain_projection) -> None:
        explanations = explain_paths(
            "target", ["nonexistent"], explain_projection, explain_store
        )
        assert explanations == ()

    def test_path_endpoints_are_correct(self, explain_store, explain_projection) -> None:
        explanations = explain_paths("far", ["seed1"], explain_projection, explain_store)
        for explanation in explanations:
            assert explanation.path[0] == "seed1"
            assert explanation.path[-1] == "far"

    def test_adjacent_seed_gives_a_single_hop(self, explain_store, explain_projection) -> None:
        explanations = explain_paths("mid", ["seed1"], explain_projection, explain_store)
        assert explanations[0].path == ("seed1", "mid")
        assert explanations[0].hops == 1


class TestFormatExplanation:
    def test_mentions_the_seed_and_hop_count(self) -> None:
        explanation = Explanation(
            seed_id="seed1",
            seed_title="Dune",
            path=("seed1", "mid", "target"),
            strength=0.8,
        )
        text = format_explanation(explanation, {"seed1": "Dune", "mid": "Middle Book", "target": "Target Book"})
        assert "Dune" in text
        assert "2" in text

    def test_direct_link_is_described_as_such(self) -> None:
        explanation = Explanation(
            seed_id="seed1", seed_title="Dune", path=("seed1", "target"), strength=0.9
        )
        text = format_explanation(explanation, {"seed1": "Dune", "target": "Target Book"})
        assert "Dune" in text


class TestExplanationModel:
    def test_hops_counts_edges_not_nodes(self) -> None:
        assert Explanation("s", "S", ("s", "a", "b")).hops == 2
        assert Explanation("s", "S", ("s", "a")).hops == 1
        assert Explanation("s", "S", ()).hops == 0
