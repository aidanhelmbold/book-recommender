"""Projection from the store into SciPy sparse matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bookmap.models import FusedEdge
from bookmap.store.projection import load_cache, project, save_cache
from tests.helpers import make_projection

PAIRS = [
    ("a", "b", 1.0),
    ("b", "c", 2.0),
    ("a", "c", 0.5),
    ("c", "d", 1.0),
]


class TestProject:
    def test_reads_the_fused_graph(self, store) -> None:
        store.replace_fused_edges(
            [FusedEdge("a", "b", 1.0), FusedEdge("b", "c", 2.0)]
        )
        projection = project(store)
        assert projection.n_nodes == 3
        assert projection.n_edges == 2
        assert set(projection.work_ids) == {"a", "b", "c"}

    def test_adjacency_is_symmetric(self, store) -> None:
        store.replace_fused_edges([FusedEdge("a", "b", 0.75)])
        projection = project(store)
        dense = projection.adjacency.toarray()
        assert np.allclose(dense, dense.T)
        i, j = projection.index["a"], projection.index["b"]
        assert dense[i, j] == pytest.approx(0.75)

    def test_index_agrees_with_work_ids(self, store) -> None:
        store.replace_fused_edges([FusedEdge("a", "b", 1.0), FusedEdge("b", "c", 1.0)])
        projection = project(store)
        for work_id, row in projection.index.items():
            assert projection.work_ids[row] == work_id

    def test_no_self_loops(self, store) -> None:
        store.replace_fused_edges([FusedEdge("a", "b", 1.0)])
        projection = project(store)
        assert np.allclose(projection.adjacency.diagonal(), 0.0)

    def test_empty_graph(self, store) -> None:
        projection = project(store)
        assert projection.n_nodes == 0
        assert projection.n_edges == 0


class TestProjectionQueries:
    def test_degree(self) -> None:
        projection = make_projection(PAIRS)
        degree = projection.degree()
        assert degree[projection.index["a"]] == 2
        assert degree[projection.index["c"]] == 3
        assert degree[projection.index["d"]] == 1

    def test_weighted_degree(self) -> None:
        """c connects to b (2.0), a (0.5), d (1.0) -> 3.5."""
        projection = make_projection(PAIRS)
        weighted = projection.weighted_degree()
        assert weighted[projection.index["c"]] == pytest.approx(3.5)

    def test_row_of(self) -> None:
        projection = make_projection(PAIRS)
        assert projection.work_ids[projection.row_of("b")] == "b"

    def test_row_of_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            make_projection(PAIRS).row_of("nope")

    def test_neighbors_of(self) -> None:
        projection = make_projection(PAIRS)
        rows, weights = projection.neighbors_of("c")
        got = {projection.work_ids[r]: w for r, w in zip(rows, weights, strict=True)}
        assert got == {
            "a": pytest.approx(0.5),
            "b": pytest.approx(2.0),
            "d": pytest.approx(1.0),
        }

    def test_subgraph_keeps_only_internal_edges(self) -> None:
        projection = make_projection(PAIRS)
        sub = projection.subgraph(["a", "b", "c"])
        assert sub.n_nodes == 3
        assert sub.n_edges == 3
        assert "d" not in sub.index

    def test_subgraph_preserves_weights(self) -> None:
        projection = make_projection(PAIRS)
        sub = projection.subgraph(["b", "c"])
        dense = sub.adjacency.toarray()
        assert dense[sub.index["b"], sub.index["c"]] == pytest.approx(2.0)

    def test_subgraph_ignores_unknown_ids(self) -> None:
        projection = make_projection(PAIRS)
        sub = projection.subgraph(["a", "b", "ghost"])
        assert sub.n_nodes == 2

    def test_to_networkx(self) -> None:
        projection = make_projection(PAIRS)
        graph = projection.to_networkx()
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 4
        assert graph["b"]["c"]["weight"] == pytest.approx(2.0)


class TestCache:
    def test_round_trips(self, tmp_path: Path) -> None:
        projection = make_projection(PAIRS)
        path = tmp_path / "graph.npz"
        save_cache(projection, path)
        loaded = load_cache(path)
        assert loaded.work_ids == projection.work_ids
        assert loaded.index == projection.index
        assert np.allclose(
            loaded.adjacency.toarray(), projection.adjacency.toarray()
        )

    def test_cache_survives_reordering_expectations(self, tmp_path: Path) -> None:
        """Row order must be preserved exactly, or cached scores map to wrong books."""
        projection = make_projection(PAIRS)
        path = tmp_path / "graph.npz"
        save_cache(projection, path)
        loaded = load_cache(path)
        for work_id, row in projection.index.items():
            assert loaded.work_ids[row] == work_id
