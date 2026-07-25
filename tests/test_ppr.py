"""Personalized PageRank, checked against NetworkX.

NetworkX is an independent implementation, so agreement to 1e-6 is real
evidence. Note the convention shift: NetworkX's ``alpha`` is the damping factor
(``1 - restart``), ours is the restart probability.
"""

from __future__ import annotations

import numpy as np
import pytest

from bookmap.graph.ppr import hub_damped_scores, personalized_pagerank, row_normalize
from tests.helpers import csr_from_pairs, networkx_ppr

LINE = [("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0)]
TRIANGLE_PLUS_TAIL = [
    ("a", "b", 1.0),
    ("b", "c", 1.0),
    ("a", "c", 1.0),
    ("c", "d", 2.0),
    ("d", "e", 0.5),
]


class TestRowNormalize:
    def test_rows_sum_to_one(self) -> None:
        adjacency, _, _ = csr_from_pairs(TRIANGLE_PLUS_TAIL)
        transition = row_normalize(adjacency)
        sums = np.asarray(transition.sum(axis=1)).ravel()
        assert np.allclose(sums, 1.0)

    def test_weights_are_respected(self) -> None:
        """c connects to a (1.0), b (1.0), d (2.0) -> d should get half of c's mass."""
        adjacency, work_ids, index = csr_from_pairs(TRIANGLE_PLUS_TAIL)
        transition = row_normalize(adjacency).toarray()
        assert transition[index["c"], index["d"]] == pytest.approx(0.5)
        assert transition[index["c"], index["a"]] == pytest.approx(0.25)

    def test_isolated_row_stays_zero(self) -> None:
        adjacency, _, index = csr_from_pairs(LINE, nodes=["a", "b", "c", "d", "lonely"])
        transition = row_normalize(adjacency)
        row = transition[index["lonely"]].toarray().ravel()
        assert np.allclose(row, 0.0)


class TestPersonalizedPageRank:
    @pytest.mark.parametrize("seeds", [["a"], ["a", "d"], ["b", "c"]])
    def test_matches_networkx(self, seeds: list[str]) -> None:
        adjacency, work_ids, index = csr_from_pairs(TRIANGLE_PLUS_TAIL)
        got = personalized_pagerank(
            adjacency, [index[s] for s in seeds], alpha=0.15, tol=1e-12, max_iter=500
        )
        expected = networkx_ppr(TRIANGLE_PLUS_TAIL, seeds, restart_alpha=0.15)
        for name, row in index.items():
            assert got[row] == pytest.approx(expected[name], abs=1e-6)

    @pytest.mark.parametrize("alpha", [0.05, 0.15, 0.5])
    def test_matches_networkx_across_alpha(self, alpha: float) -> None:
        adjacency, _, index = csr_from_pairs(LINE)
        got = personalized_pagerank(
            adjacency, [index["a"]], alpha=alpha, tol=1e-12, max_iter=500
        )
        expected = networkx_ppr(LINE, ["a"], restart_alpha=alpha)
        for name, row in index.items():
            assert got[row] == pytest.approx(expected[name], abs=1e-6)

    def test_is_a_probability_distribution(self) -> None:
        adjacency, _, index = csr_from_pairs(TRIANGLE_PLUS_TAIL)
        ppr = personalized_pagerank(adjacency, [index["a"]])
        assert ppr.sum() == pytest.approx(1.0, abs=1e-9)
        assert (ppr >= 0).all()

    def test_conserves_mass_with_isolated_nodes(self) -> None:
        """Dangling/isolated nodes must not leak probability."""
        adjacency, _, index = csr_from_pairs(
            LINE, nodes=["a", "b", "c", "d", "lonely1", "lonely2"]
        )
        ppr = personalized_pagerank(adjacency, [index["a"]])
        assert ppr.sum() == pytest.approx(1.0, abs=1e-9)

    def test_seed_mass_decays_with_distance(self) -> None:
        adjacency, _, index = csr_from_pairs(LINE)
        ppr = personalized_pagerank(adjacency, [index["a"]])
        assert ppr[index["a"]] > ppr[index["b"]] > ppr[index["c"]] > ppr[index["d"]]

    def test_reachable_from_both_seeds_beats_reachable_from_one(self) -> None:
        """The whole point of multi-seed PPR: score against the *set*.

        ``both`` and ``one`` each have degree 2, so degree cannot explain the
        gap -- the only difference is that ``both`` is adjacent to two seeds and
        ``one`` to a single seed. A recommender that scored each seed separately
        and merged the lists would rank these two equally.
        """
        pairs = [
            ("s1", "both", 1.0),
            ("s2", "both", 1.0),
            ("s1", "one", 1.0),
            ("one", "filler", 1.0),
        ]
        adjacency, _, index = csr_from_pairs(pairs)
        ppr = personalized_pagerank(adjacency, [index["s1"], index["s2"]])
        assert ppr[index["both"]] > ppr[index["one"]]

    def test_adding_a_seed_lifts_books_near_it(self) -> None:
        """Extending the seed set must pull its neighbourhood up."""
        pairs = [
            ("s1", "x", 1.0),
            ("x", "y", 1.0),
            ("s2", "y", 1.0),
            ("s2", "z", 1.0),
        ]
        adjacency, _, index = csr_from_pairs(pairs)
        one_seed = personalized_pagerank(adjacency, [index["s1"]])
        two_seeds = personalized_pagerank(adjacency, [index["s1"], index["s2"]])
        assert two_seeds[index["z"]] > one_seed[index["z"]]

    def test_seed_weights_shift_the_distribution(self) -> None:
        adjacency, _, index = csr_from_pairs(LINE)
        rows = [index["a"], index["d"]]
        lopsided = personalized_pagerank(
            adjacency, rows, seed_weights=np.array([0.9, 0.1])
        )
        even = personalized_pagerank(adjacency, rows)
        assert lopsided[index["a"]] > even[index["a"]]
        assert lopsided[index["d"]] < even[index["d"]]

    def test_rejects_empty_seed_set(self) -> None:
        adjacency, _, _ = csr_from_pairs(LINE)
        with pytest.raises(ValueError):
            personalized_pagerank(adjacency, [])

    def test_converges_well_before_max_iter(self) -> None:
        adjacency, _, index = csr_from_pairs(TRIANGLE_PLUS_TAIL)
        tight = personalized_pagerank(adjacency, [index["a"]], tol=1e-12, max_iter=500)
        loose = personalized_pagerank(adjacency, [index["a"]], tol=1e-10, max_iter=60)
        assert np.allclose(tight, loose, atol=1e-8)


class TestHubDamping:
    def test_penalises_the_hub(self) -> None:
        """Equal PPR mass, wildly different degree -> the specific book wins.

        ppr 0.4 / 100**0.5 = 0.04 versus 0.4 / 4**0.5 = 0.2.
        """
        ppr = np.array([0.4, 0.4])
        degree = np.array([100.0, 4.0])
        scores = hub_damped_scores(ppr, degree, beta=0.5)
        assert scores[0] == pytest.approx(0.04)
        assert scores[1] == pytest.approx(0.2)
        assert scores[1] > scores[0]

    def test_beta_zero_is_a_no_op(self) -> None:
        ppr = np.array([0.4, 0.1, 0.5])
        degree = np.array([1000.0, 2.0, 30.0])
        assert np.allclose(hub_damped_scores(ppr, degree, beta=0.0), ppr)

    def test_zero_degree_does_not_divide_by_zero(self) -> None:
        scores = hub_damped_scores(np.array([0.0, 0.5]), np.array([0.0, 4.0]), beta=0.25)
        assert np.isfinite(scores).all()
        assert scores[0] == pytest.approx(0.0)

    def test_default_beta_still_lets_a_strong_hub_win(self) -> None:
        """Damping must temper popularity, not invert it.

        A hub with 20x the PPR mass of an obscure book should still rank first at
        the default beta; if it does not, the damping is too aggressive and the
        recommender will only ever surface obscurities.
        """
        ppr = np.array([0.40, 0.02])
        degree = np.array([500.0, 3.0])
        scores = hub_damped_scores(ppr, degree, beta=0.25)
        assert scores[0] > scores[1]
