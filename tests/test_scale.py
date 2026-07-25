"""Scale sanity checks, marked slow.

The real Goodreads graph is ~2.4M nodes and tens of millions of edges. These
tests exist so a change that quietly makes the pipeline quadratic is caught here
rather than discovered on a 2GB dump. Run with ``uv run pytest -m slow``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from bookmap.graph.ppr import hub_damped_scores, personalized_pagerank, row_normalize

pytestmark = pytest.mark.slow

N_NODES = 500_000
AVG_DEGREE = 20


@pytest.fixture(scope="module")
def big_graph() -> sp.csr_matrix:
    """A synthetic power-law-ish graph, built once for the module."""
    rng = np.random.default_rng(1917)
    n_edges = N_NODES * AVG_DEGREE // 2
    # Preferential-attachment-flavoured: bias one endpoint toward low indices so
    # the degree distribution is skewed like a real co-purchase graph.
    src = rng.integers(0, N_NODES, size=n_edges)
    dst = (rng.power(0.3, size=n_edges) * N_NODES).astype(np.int64)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    data = np.ones(len(src))
    adjacency = sp.csr_matrix(
        (np.concatenate([data, data]), (np.concatenate([src, dst]), np.concatenate([dst, src]))),
        shape=(N_NODES, N_NODES),
    )
    adjacency.sum_duplicates()
    return adjacency


class TestPPRAtScale:
    def test_completes_quickly(self, big_graph: sp.csr_matrix) -> None:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, N_NODES, size=5).tolist()
        start = time.perf_counter()
        ppr = personalized_pagerank(big_graph, seeds, tol=1e-8, max_iter=100)
        elapsed = time.perf_counter() - start
        assert ppr.shape == (N_NODES,)
        assert elapsed < 30.0, f"PPR took {elapsed:.1f}s on {N_NODES} nodes"

    def test_still_a_probability_distribution(self, big_graph: sp.csr_matrix) -> None:
        ppr = personalized_pagerank(big_graph, [0, 1, 2], tol=1e-8)
        assert abs(ppr.sum() - 1.0) < 1e-6
        assert (ppr >= 0).all()

    def test_row_normalize_does_not_densify(self, big_graph: sp.csr_matrix) -> None:
        """Densifying a 500k-node graph would need ~2TB of RAM.

        This catches the classic mistake of dividing by a dense column vector.
        """
        transition = row_normalize(big_graph)
        assert sp.issparse(transition)
        assert transition.nnz <= big_graph.nnz

    def test_hub_damping_is_vectorised(self, big_graph: sp.csr_matrix) -> None:
        ppr = personalized_pagerank(big_graph, [0], max_iter=20)
        degree = np.asarray((big_graph > 0).sum(axis=1)).ravel().astype(float)
        start = time.perf_counter()
        scores = hub_damped_scores(ppr, degree, beta=0.25)
        assert time.perf_counter() - start < 1.0
        assert np.isfinite(scores).all()


class TestProjectionAtScale:
    def test_arrow_projection_is_fast(self, tmp_path) -> None:
        """The Arrow path is why DuckDB was chosen over SQLite.

        Two million edges must project in seconds; a per-row Python cursor would
        take a minute or more, on every single build.
        """
        from bookmap.models import FusedEdge
        from bookmap.store.db import Store
        from bookmap.store.projection import project

        n_edges = 2_000_000
        rng = np.random.default_rng(7)
        src = rng.integers(0, 200_000, size=n_edges)
        dst = rng.integers(0, 200_000, size=n_edges)

        with Store.open(tmp_path / "big.duckdb") as store:
            store.replace_fused_edges(
                FusedEdge(f"w{a}", f"w{b}", 0.5)
                for a, b in zip(src, dst, strict=True)
                if a != b
            )
            start = time.perf_counter()
            projection = project(store)
            elapsed = time.perf_counter() - start

        assert projection.n_nodes > 0
        assert elapsed < 60.0, f"projection took {elapsed:.1f}s"
