"""Scale sanity checks, marked slow.

The real Goodreads graph is ~2.4M nodes and tens of millions of edges. These
tests exist so a change that quietly makes the pipeline quadratic is caught here
rather than discovered on a 2GB dump. Run with ``uv run pytest -m slow``.
"""

from __future__ import annotations

import resource
import time

import numpy as np
import pytest
import scipy.sparse as sp

from bookmap.graph.ppr import hub_damped_scores, personalized_pagerank, row_normalize

pytestmark = pytest.mark.slow

N_NODES = 500_000
AVG_DEGREE = 20

# Ref resolution at dump scale. The real UCSD Goodreads dump is ~2.36M books and
# order 40M similar-books edges, every endpoint of which arrives as a namespaced
# ref; these numbers are half that, which is enough to catch anything that scales
# with the *product* of table sizes rather than their sum.
REF_BOOKS = 2_000_000
REFS_PER_BOOK = 10
REF_ID_SPACE = 2_222_222
"""Target ids are drawn from a space 11% larger than the book table.

That is the dump's most consequential quirk: ``similar_books`` cites books
outside the corpus constantly, so ~10% of endpoints resolve to nothing and their
edges have to be dropped rather than kept as titleless phantom nodes.
"""

RESOLVE_BUDGET_SECONDS = 120.0
RESOLVE_RSS_BUDGET_BYTES = 8 * 1024**3
_STRIDE = 104729  # prime, so the targets of one source book cannot collide


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


def _synthetic_refs() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(src_id, dst_id, rank, unresolvable)`` for the synthetic edge set.

    Built with array arithmetic rather than a loop: 20M rows through a Python
    ``for`` would cost more than the query under test. Each source book cites
    ``REFS_PER_BOOK`` targets at fixed prime strides, which makes every
    ``(src, dst)`` pair unique and never a self-loop -- so the expected surviving
    row count is exactly "all of them minus the unresolvable ones", with nothing
    lost to de-duplication that would blur the assertion.
    """
    src = np.repeat(np.arange(REF_BOOKS, dtype=np.int64), REFS_PER_BOOK)
    strides = np.arange(1, REFS_PER_BOOK + 1, dtype=np.int64) * _STRIDE
    dst = (src + np.tile(strides, REF_BOOKS)) % REF_ID_SPACE
    rank = np.tile(np.arange(1, REFS_PER_BOOK + 1, dtype=np.int64), REF_BOOKS)
    return src, dst, rank, int((dst >= REF_BOOKS).sum())


@pytest.fixture(scope="module")
def resolved_refs(tmp_path_factory) -> dict[str, object]:
    """Build a dump-scale ``edges_raw`` of namespaced refs, then resolve it once.

    Resolution is timed here rather than in each test because both the build and
    the resolve are minutes of work; the tests below assert on the recorded
    outcome.
    """
    import pyarrow as pa

    from bookmap.models import EdgeKind, SourceName
    from bookmap.sources.base import unresolved_ref
    from bookmap.store.db import Store

    directory = tmp_path_factory.mktemp("refscale")
    src, dst, rank, unresolvable = _synthetic_refs()
    ref_prefix = unresolved_ref(SourceName.GOODREADS_UCSD, "\x00").removesuffix("\x00")

    with Store.open(
        directory / "refs.duckdb",
        memory_limit="6GB",
        temp_directory=directory / "spill",
        # A cap rather than DuckDB's default of 90% of the volume: a spill that
        # fills the disk is what this whole test exists to prevent, and it should
        # surface as a failed query, not a dead machine.
        max_temp_directory_size="12GB",
    ) as store:
        conn = store.conn
        with store.bulk_load():
            conn.execute(
                "INSERT INTO books (work_id, title, goodreads_id, title_norm) "
                "SELECT 'gr-' || CAST(i AS VARCHAR), 'Book ' || CAST(i AS VARCHAR), "
                "CAST(i AS VARCHAR), 'book ' || CAST(i AS VARCHAR) FROM range(?) t(i)",
                [REF_BOOKS],
            )
            conn.execute(
                "INSERT INTO aliases (id_type, id_value, work_id) "
                "SELECT 'goodreads_id', CAST(i AS VARCHAR), 'gr-' || CAST(i AS VARCHAR) "
                "FROM range(?) t(i)",
                [REF_BOOKS],
            )

            # Chunked only because one statement holds its whole primary-key
            # index delta in memory before committing; the rows themselves still
            # cross into DuckDB as Arrow buffers, never as Python tuples.
            chunk = 4_000_000
            for start in range(0, len(src), chunk):
                stop = start + chunk
                batch = pa.table(
                    {
                        "src": pa.array(src[start:stop]),
                        "dst": pa.array(dst[start:stop]),
                        "rank": pa.array(rank[start:stop]),
                    }
                )
                conn.register("_scale_edges", batch)
                try:
                    conn.execute(
                        "INSERT INTO edges_raw (src, dst, kind, rank, source) "
                        "SELECT ? || CAST(src AS VARCHAR), ? || CAST(dst AS VARCHAR), "
                        "?, rank, ? FROM _scale_edges",
                        [
                            ref_prefix,
                            ref_prefix,
                            EdgeKind.GR_SIMILAR.value,
                            SourceName.GOODREADS_UCSD.value,
                        ],
                    )
                finally:
                    conn.unregister("_scale_edges")

        before = int(conn.execute("SELECT count(*) FROM edges_raw").fetchone()[0])
        del src, dst, rank

        start_time = time.perf_counter()
        dropped = store.resolve_refs()
        elapsed = time.perf_counter() - start_time

        repeat_start = time.perf_counter()
        dropped_again = store.resolve_refs()
        repeat_elapsed = time.perf_counter() - repeat_start

        namespaced = int(
            conn.execute(
                "SELECT count(*) FROM edges_raw WHERE strpos(src, ':') > 0 OR strpos(dst, ':') > 0"
            ).fetchone()[0]
        )
        # Distinct endpoints first: 18M rows anti-joined against the book table
        # would answer the same question the expensive way.
        unknown = int(
            conn.execute(
                """
                SELECT count(*) FROM (
                    SELECT DISTINCT endpoint FROM (
                        SELECT src AS endpoint FROM edges_raw
                        UNION ALL
                        SELECT dst AS endpoint FROM edges_raw
                    )
                ) e
                WHERE NOT EXISTS (SELECT 1 FROM books b WHERE b.work_id = e.endpoint)
                """
            ).fetchone()[0]
        )
        kept = int(conn.execute("SELECT count(*) FROM edges_raw").fetchone()[0])
        spilled = sorted(p.name for p in (directory / "spill").glob("**/*") if p.is_file())

    return {
        "before": before,
        "kept": kept,
        "dropped": dropped,
        "dropped_again": dropped_again,
        "expected_dropped": unresolvable,
        "elapsed": elapsed,
        "repeat_elapsed": repeat_elapsed,
        "namespaced": namespaced,
        "unknown": unknown,
        "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "spilled": spilled,
        "db_bytes": (directory / "refs.duckdb").stat().st_size,
    }


class TestRefResolutionAtScale:
    """Ref resolution over a dump-shaped ``edges_raw``.

    This is the test that was missing when a six-way join over 40M rows -- four
    of whose join keys were ``substr``/``split_part`` expressions, so no index
    could serve them -- shipped green and then hung on a user's 9.2GB dump before
    filling their disk with spill files.
    """

    def test_completes_within_budget(self, resolved_refs: dict[str, object]) -> None:
        assert resolved_refs["elapsed"] < RESOLVE_BUDGET_SECONDS, (
            f"resolve_refs took {resolved_refs['elapsed']:.1f}s "
            f"on {resolved_refs['before']:,} ref edges"
        )

    def test_drops_exactly_the_unresolvable_edges(self, resolved_refs: dict[str, object]) -> None:
        """~10% of targets name books outside the corpus; those edges are not real."""
        assert resolved_refs["dropped"] == resolved_refs["expected_dropped"]
        assert resolved_refs["kept"] == (
            resolved_refs["before"] - resolved_refs["expected_dropped"]
        )

    def test_surviving_endpoints_are_canonical_work_ids(
        self, resolved_refs: dict[str, object]
    ) -> None:
        assert resolved_refs["namespaced"] == 0
        assert resolved_refs["unknown"] == 0

    def test_second_pass_is_a_cheap_no_op(self, resolved_refs: dict[str, object]) -> None:
        """Idempotence, and cheaply: nothing is left namespaced, so nothing is rebuilt."""
        assert resolved_refs["dropped_again"] == 0
        assert resolved_refs["repeat_elapsed"] < RESOLVE_BUDGET_SECONDS / 4, (
            f"a no-op second pass took {resolved_refs['repeat_elapsed']:.1f}s"
        )

    def test_memory_stays_bounded(self, resolved_refs: dict[str, object]) -> None:
        """The old implementation materialised every rewritten row at once.

        A 40M-row result set held in memory is what turned into gigabytes of
        spill; the fix streams through a rebuild whose peak is a hash table over
        *distinct refs*, not over edges.
        """
        assert resolved_refs["peak_rss"] < RESOLVE_RSS_BUDGET_BYTES, (
            f"peak RSS {resolved_refs['peak_rss'] / 1024**3:.2f}GiB"
        )


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
        # Canonicalised to src < dst, matching what fuse_edges emits and what the
        # schema documents. Storing both (x, y) and (y, x) would make project()
        # sum the two directions and double those weights -- the timing assertion
        # below would still pass, so the invariant has to be honoured here rather
        # than defended against downstream.
        low = np.minimum(src, dst)
        high = np.maximum(src, dst)

        with Store.open(tmp_path / "big.duckdb") as store:
            store.replace_fused_edges(
                FusedEdge(f"w{a}", f"w{b}", 0.5)
                for a, b in zip(low, high, strict=True)
                if a != b
            )
            start = time.perf_counter()
            projection = project(store)
            elapsed = time.perf_counter() - start

        assert projection.n_nodes > 0
        assert elapsed < 60.0, f"projection took {elapsed:.1f}s"
