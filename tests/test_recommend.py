"""The recommendation pipeline.

The MMR case below is arithmetic, not judgement: the scores and neighbourhoods
are chosen so the correct output order can be derived on paper, and so that the
order *changes* when diversity is switched off. A diversity pass that made no
difference would otherwise pass a weaker test silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from bookmap.config import RecommendConfig
from bookmap.models import Book, Edge, EdgeKind, SourceName
from bookmap.recommend import mmr_rerank, recommend, resolve_seeds
from tests.helpers import make_projection

# A and B are redundant (identical neighbourhoods); C is distinct.
MMR_PAIRS = [
    ("A", "h1", 1.0),
    ("A", "h2", 1.0),
    ("B", "h1", 1.0),
    ("B", "h2", 1.0),
    ("C", "h3", 1.0),
    ("C", "h4", 1.0),
]


def _mmr_setup():
    projection = make_projection(MMR_PAIRS)
    scores = np.zeros(projection.n_nodes)
    scores[projection.index["A"]] = 1.0
    scores[projection.index["B"]] = 0.9
    scores[projection.index["C"]] = 0.6
    rows = [projection.index[n] for n in ("A", "B", "C")]
    return projection, scores, rows


class TestMMR:
    def test_diversity_promotes_the_distinct_candidate(self) -> None:
        """Hand-computed at lambda=0.7.

        Pick 1: A wins on 0.7*1.0 = 0.70 (no penalty yet).
        Pick 2: B scores 0.7*0.9 - 0.3*jaccard({h1,h2},{h1,h2}) = 0.63 - 0.30 = 0.33
                C scores 0.7*0.6 - 0.3*jaccard({h3,h4},{h1,h2}) = 0.42 - 0.00 = 0.42
                -> C, despite B having the higher raw score.
        """
        projection, scores, rows = _mmr_setup()
        got = mmr_rerank(rows, scores, projection, n=3, lambda_=0.7)
        assert [projection.work_ids[r] for r in got] == ["A", "C", "B"]

    def test_lambda_one_falls_back_to_pure_score_order(self) -> None:
        projection, scores, rows = _mmr_setup()
        got = mmr_rerank(rows, scores, projection, n=3, lambda_=1.0)
        assert [projection.work_ids[r] for r in got] == ["A", "B", "C"]

    def test_respects_n(self) -> None:
        projection, scores, rows = _mmr_setup()
        assert len(mmr_rerank(rows, scores, projection, n=2, lambda_=0.7)) == 2

    def test_never_repeats_a_candidate(self) -> None:
        projection, scores, rows = _mmr_setup()
        got = mmr_rerank(rows, scores, projection, n=3, lambda_=0.7)
        assert len(set(got)) == len(got)

    def test_n_larger_than_candidates(self) -> None:
        projection, scores, rows = _mmr_setup()
        assert len(mmr_rerank(rows, scores, projection, n=99, lambda_=0.7)) == 3

    def test_empty_candidates(self) -> None:
        projection, scores, _ = _mmr_setup()
        assert mmr_rerank([], scores, projection, n=5) == []


@pytest.fixture
def recommend_store(store):
    """A small library with two genre clusters and a shared-author pair.

    Deliberately includes a high-degree hub ("Popular Everything") wired to every
    node, so hub damping has something real to suppress.
    """
    books = [
        Book(work_id="w-dune", title="Dune", authors=("Frank Herbert",)),
        Book(work_id="w-messiah", title="Dune Messiah", authors=("Frank Herbert",)),
        Book(work_id="w-foundation", title="Foundation", authors=("Isaac Asimov",)),
        Book(work_id="w-hyperion", title="Hyperion", authors=("Dan Simmons",)),
        Book(work_id="w-blindsight", title="Blindsight", authors=("Peter Watts",)),
        Book(work_id="w-emma", title="Emma", authors=("Jane Austen",)),
        Book(work_id="w-persuasion", title="Persuasion", authors=("Jane Austen",)),
        Book(work_id="w-hub", title="Popular Everything", authors=("Anon",)),
    ]
    sf = ["w-dune", "w-messiah", "w-foundation", "w-hyperion", "w-blindsight"]
    lit = ["w-emma", "w-persuasion"]
    edges: list[Edge] = []
    for group in (sf, lit):
        for i, a in enumerate(group):
            for rank, b in enumerate(group[i + 1 :], start=1):
                edges.append(Edge(a, b, EdgeKind.GR_SIMILAR, rank, SourceName.DEMO))
                edges.append(Edge(b, a, EdgeKind.GR_SIMILAR, rank, SourceName.DEMO))
    for node in sf + lit:
        edges.append(Edge("w-hub", node, EdgeKind.AZ_ALSO_BOUGHT, 1, SourceName.DEMO))
        edges.append(Edge(node, "w-hub", EdgeKind.AZ_ALSO_BOUGHT, 1, SourceName.DEMO))

    from bookmap.graph.build import fuse_edges

    store.upsert_books(books)
    store.insert_raw_edges(edges)
    store.replace_fused_edges(fuse_edges(edges))
    return store


@pytest.fixture
def recommend_projection(recommend_store):
    from bookmap.store.projection import project

    return project(recommend_store)


class TestResolveSeeds:
    def test_resolves_known_titles(self, recommend_store) -> None:
        matches, unresolved = resolve_seeds(["Dune", "Foundation"], recommend_store)
        assert {m.work_id for m in matches} == {"w-dune", "w-foundation"}
        assert unresolved == []

    def test_reports_unknown_titles(self, recommend_store) -> None:
        matches, unresolved = resolve_seeds(["Dune", "Zzzqqx Nonexistent"], recommend_store)
        assert [m.work_id for m in matches] == ["w-dune"]
        assert unresolved == ["Zzzqqx Nonexistent"]

    def test_empty_query_list(self, recommend_store) -> None:
        assert resolve_seeds([], recommend_store) == ([], [])


class TestRecommend:
    def test_never_recommends_a_seed(self, recommend_store, recommend_projection) -> None:
        """The most obvious possible failure mode."""
        result = recommend(
            ["Dune", "Foundation"], recommend_store, recommend_projection, n=10
        )
        got = {r.work_id for r in result.recommendations}
        assert "w-dune" not in got
        assert "w-foundation" not in got

    def test_returns_neighbours_of_the_seed_cluster(
        self, recommend_store, recommend_projection
    ) -> None:
        result = recommend(
            ["Dune", "Foundation"], recommend_store, recommend_projection, n=3
        )
        got = {r.work_id for r in result.recommendations}
        assert got & {"w-hyperion", "w-blindsight", "w-messiah"}

    def test_respects_n(self, recommend_store, recommend_projection) -> None:
        result = recommend(["Dune"], recommend_store, recommend_projection, n=2)
        assert len(result.recommendations) == 2

    def test_surfaces_unresolved_seeds(self, recommend_store, recommend_projection) -> None:
        result = recommend(
            ["Dune", "Zzzqqx Nonexistent"], recommend_store, recommend_projection, n=5
        )
        assert result.unresolved == ("Zzzqqx Nonexistent",)
        assert result.recommendations

    def test_all_seeds_unresolved_yields_no_recommendations(
        self, recommend_store, recommend_projection
    ) -> None:
        result = recommend(["Zzzqqx"], recommend_store, recommend_projection, n=5)
        assert result.recommendations == ()
        assert result.unresolved == ("Zzzqqx",)

    def test_recommendations_are_sorted_by_score(
        self, recommend_store, recommend_projection
    ) -> None:
        result = recommend(["Dune"], recommend_store, recommend_projection, n=5)
        scores = [r.score for r in result.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_titles_are_populated(self, recommend_store, recommend_projection) -> None:
        result = recommend(["Dune"], recommend_store, recommend_projection, n=3)
        assert all(r.title for r in result.recommendations)

    def test_hub_damping_demotes_the_hub(
        self, recommend_store, recommend_projection
    ) -> None:
        """"Popular Everything" is adjacent to every book.

        Undamped it should rank at or near the top for any seed; damped it should
        fall. If this test fails, hub damping is not wired into the pipeline.
        """
        undamped = recommend(
            ["Dune"],
            recommend_store,
            recommend_projection,
            n=8,
            config=RecommendConfig(hub_beta=0.0, mmr_lambda=1.0),
        )
        damped = recommend(
            ["Dune"],
            recommend_store,
            recommend_projection,
            n=8,
            config=RecommendConfig(hub_beta=0.9, mmr_lambda=1.0),
        )

        def rank_of(result, work_id: str) -> int:
            ids = [r.work_id for r in result.recommendations]
            return ids.index(work_id) if work_id in ids else len(ids)

        assert rank_of(damped, "w-hub") > rank_of(undamped, "w-hub")

    def test_exclude_same_author_drops_the_series(
        self, recommend_store, recommend_projection
    ) -> None:
        """Seeding Dune should not just return more Herbert when this is on."""
        result = recommend(
            ["Dune"],
            recommend_store,
            recommend_projection,
            n=8,
            config=RecommendConfig(exclude_same_author=True),
        )
        assert "w-messiah" not in {r.work_id for r in result.recommendations}

    def test_same_author_included_by_default(
        self, recommend_store, recommend_projection
    ) -> None:
        result = recommend(["Dune"], recommend_store, recommend_projection, n=8)
        assert "w-messiah" in {r.work_id for r in result.recommendations}

    def test_explanations_terminate_at_a_seed(
        self, recommend_store, recommend_projection
    ) -> None:
        result = recommend(
            ["Dune", "Emma"], recommend_store, recommend_projection, n=5
        )
        seed_ids = {m.work_id for m in result.seeds}
        for rec in result.recommendations:
            assert rec.explanations, f"no explanation for {rec.work_id}"
            for explanation in rec.explanations:
                assert explanation.path[0] in seed_ids
                assert explanation.path[-1] == rec.work_id

    def test_seeds_are_echoed_in_the_result(
        self, recommend_store, recommend_projection
    ) -> None:
        result = recommend(["Dune"], recommend_store, recommend_projection, n=3)
        assert {m.work_id for m in result.seeds} == {"w-dune"}

    def test_degree_and_raw_ppr_are_reported(
        self, recommend_store, recommend_projection
    ) -> None:
        """Needed to debug a surprising ranking without re-running the pipeline."""
        result = recommend(["Dune"], recommend_store, recommend_projection, n=3)
        for rec in result.recommendations:
            assert rec.degree > 0
            assert rec.raw_ppr > 0
