"""End-to-end behaviour over the bundled demo corpus.

These are the tests that say whether the thing actually works as a book
recommender, rather than whether each part is individually correct. They assert
recommendation *quality*, which is why the demo corpus is built with known genre
clusters -- seeding hard SF and getting a romance novel back is a real failure,
and nothing in the unit tests would catch it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from bookmap.config import RecommendConfig
from bookmap.graph.build import fuse_edges
from bookmap.graph.communities import (
    community_sizes,
    detect_communities,
    label_communities,
)
from bookmap.recommend import recommend
from bookmap.sources.demo import DemoSource
from bookmap.store.db import Store
from bookmap.store.projection import project


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """Ingest and build the demo corpus once for the whole module.

    This must mirror everything ``bookmap build`` does, community detection
    included: several tests below read communities back through
    ``store.node_community``, and a fixture that only fuses edges leaves that
    column NULL, so those assertions would compare None against None and pass
    for the wrong reason.
    """
    db = tmp_path_factory.mktemp("integration") / "demo.duckdb"
    source = DemoSource()
    with Store.open(db) as store:
        store.upsert_books(source.iter_books())
        store.insert_raw_edges(source.iter_edges())
        store.resolve_refs()
        store.replace_fused_edges(fuse_edges(store.iter_raw_edges()))

        projection = project(store)
        labels = detect_communities(projection)
        degree = projection.degree()
        weighted = projection.weighted_degree()
        store.write_node_metrics(
            [
                (work_id, int(degree[row]), float(weighted[row]), int(labels[row]), None)
                for row, work_id in enumerate(projection.work_ids)
            ]
        )
        books = store.get_books(projection.work_ids)
        titles = [
            books[work_id].title if work_id in books else work_id
            for work_id in projection.work_ids
        ]
        sizes = community_sizes(labels)
        store.write_communities(
            [
                (community, label, sizes[community])
                for community, label in label_communities(labels, titles).items()
            ]
        )
    return db


@pytest.fixture
def store(built: Path):
    with Store.open(built, read_only=True) as s:
        yield s


@pytest.fixture
def projection(store):
    return project(store)


class TestGraphShape:
    def test_graph_is_connected_enough_to_traverse(self, projection) -> None:
        """A graph of isolated pairs cannot support multi-hop recommendation."""
        import networkx as nx

        graph = projection.to_networkx()
        largest = max(nx.connected_components(graph), key=len)
        assert len(largest) > 0.7 * projection.n_nodes

    def test_degree_distribution_is_skewed(self, projection) -> None:
        """Real book graphs are hub-heavy. A uniform corpus would make hub
        damping untestable and the demo unrepresentative."""
        degree = projection.degree()
        assert degree.max() > 3 * degree.mean()

    def test_has_multiple_communities(self, projection) -> None:
        labels = detect_communities(projection)
        assert len(set(labels.tolist())) >= 3


class TestRecommendationQuality:
    def test_science_fiction_seeds_return_science_fiction(self, store, projection) -> None:
        """The headline behaviour. Seeds are three hard-SF classics; the results
        must stay in that neighbourhood rather than wandering the corpus."""
        result = recommend(
            ["Dune", "Foundation", "Hyperion"], store, projection, n=10
        )
        assert len(result.recommendations) == 10
        seed_communities = {
            store.node_community(m.work_id) for m in result.seeds
        }
        rec_communities = [
            store.node_community(r.work_id) for r in result.recommendations
        ]
        overlap = sum(1 for c in rec_communities if c in seed_communities)
        assert overlap >= 5

    def test_no_seed_is_recommended(self, store, projection) -> None:
        result = recommend(["Dune", "Foundation", "Hyperion"], store, projection, n=20)
        seed_ids = {m.work_id for m in result.seeds}
        assert not (seed_ids & {r.work_id for r in result.recommendations})

    def test_every_recommendation_is_explained(self, store, projection) -> None:
        result = recommend(["Dune", "Foundation", "Hyperion"], store, projection, n=20)
        for rec in result.recommendations:
            assert rec.explanations, f"{rec.title} has no explanation"

    def test_explanation_paths_terminate_at_a_real_seed(self, store, projection) -> None:
        result = recommend(["Dune", "Foundation", "Hyperion"], store, projection, n=20)
        seed_ids = {m.work_id for m in result.seeds}
        for rec in result.recommendations:
            for explanation in rec.explanations:
                assert explanation.path[0] in seed_ids
                assert explanation.path[-1] == rec.work_id
                assert len(explanation.path) >= 2

    def test_explanation_paths_are_real_edges(self, store, projection) -> None:
        """A path that does not exist in the graph is a fabricated justification."""
        result = recommend(["Dune", "Foundation"], store, projection, n=10)
        graph = projection.to_networkx()
        for rec in result.recommendations:
            for explanation in rec.explanations:
                for a, b in zip(explanation.path, explanation.path[1:], strict=False):
                    assert graph.has_edge(a, b), f"phantom edge {a} -> {b}"

    def test_results_do_not_collapse_into_one_community(self, store, projection) -> None:
        """Mixed seeds must produce a mixed list, not just the densest region.

        Seeding hard SF and Regency romance should surface both. The failure this
        guards is collapse: every result landing in one cluster, which is
        technically relevant and practically useless.

        The bar is deliberately "spans both seeded clusters, neither crowding the
        other out" rather than a fixed count of clusters. An earlier version
        demanded three or more, which is stricter than correct -- it can only be
        met by wandering *outside* both seeded genres, and it was in fact only
        satisfiable while a scale bug in the MMR redundancy term was forcing
        pathological over-diversification.
        """
        result = recommend(
            ["Dune", "Emma"], store, projection, n=12, config=RecommendConfig()
        )
        communities = [store.node_community(r.work_id) for r in result.recommendations]
        distinct = set(communities)
        assert len(distinct) >= 2, f"collapsed into {distinct}"

        largest = max(Counter(communities).values())
        assert largest < 0.8 * len(communities), (
            f"one community holds {largest} of {len(communities)} results"
        )

    def test_mixed_seeds_pull_from_both_regions(self, store, projection) -> None:
        result = recommend(["Dune", "Emma"], store, projection, n=16)
        seed_communities = {store.node_community(m.work_id) for m in result.seeds}
        rec_communities = {store.node_community(r.work_id) for r in result.recommendations}
        assert len(seed_communities & rec_communities) >= 2

    def test_single_seed_still_works(self, store, projection) -> None:
        result = recommend(["Dune"], store, projection, n=10)
        assert len(result.recommendations) == 10

    def test_hub_damping_changes_the_result_set(self, store, projection) -> None:
        undamped = recommend(
            ["Dune"], store, projection, n=15,
            config=RecommendConfig(hub_beta=0.0, mmr_lambda=1.0),
        )
        damped = recommend(
            ["Dune"], store, projection, n=15,
            config=RecommendConfig(hub_beta=0.5, mmr_lambda=1.0),
        )
        assert [r.work_id for r in undamped.recommendations] != [
            r.work_id for r in damped.recommendations
        ]

    def test_damped_results_are_less_popular_on_average(self, store, projection) -> None:
        """The point of damping, stated as a measurement."""
        undamped = recommend(
            ["Dune"], store, projection, n=15,
            config=RecommendConfig(hub_beta=0.0, mmr_lambda=1.0),
        )
        damped = recommend(
            ["Dune"], store, projection, n=15,
            config=RecommendConfig(hub_beta=0.6, mmr_lambda=1.0),
        )
        mean_undamped = sum(r.degree for r in undamped.recommendations) / 15
        mean_damped = sum(r.degree for r in damped.recommendations) / 15
        assert mean_damped < mean_undamped

    def test_is_deterministic(self, store, projection) -> None:
        """The same query twice must give the same answer."""
        first = recommend(["Dune", "Foundation"], store, projection, n=10)
        second = recommend(["Dune", "Foundation"], store, projection, n=10)
        assert [r.work_id for r in first.recommendations] == [
            r.work_id for r in second.recommendations
        ]

    def test_fuzzy_seed_titles_resolve(self, store, projection) -> None:
        """Users mistype. "Dune" typed as "dune " must still work."""
        result = recommend(["dune ", "foundaton"], store, projection, n=5)
        assert result.recommendations
        assert len(result.unresolved) <= 1
