"""``build --min-ratings`` drops the long tail, and must not drop the unknown.

Obscure books carry thin, unreliable ``similar_books`` lists, so filtering them
yields a smaller graph and arguably better recommendations. The trap is in the
data: ``ratings_count`` arrives from the dump as a *string*, frequently ``""``, so
a naive coercion turns "we do not know" into "zero" and silently deletes every
book whose count is absent. On the real dump that is a large slice of the corpus,
and it would look like a working filter.

The other requirement is that filtering a book must also remove the edges that
point at it. A retained edge with a deleted endpoint is a dangling ref, and the
projection would carry a node with no book behind it.
"""

from __future__ import annotations

import pytest

from bookmap.config import GraphConfig
from bookmap.graph.build import fuse_in_sql
from bookmap.models import Book, Edge, EdgeKind, SourceName


@pytest.fixture
def rated_store(store):
    """Four books spanning every case the real dump produces."""
    store.upsert_books(
        [
            Book(work_id="w-big", title="Bestseller", authors=("A",), ratings_count=500_000),
            Book(work_id="w-mid", title="Midlist", authors=("B",), ratings_count=800),
            Book(work_id="w-tiny", title="Obscure", authors=("C",), ratings_count=3),
            # The case that matters: the dump gave "" and the adapter coerced it
            # to None rather than 0.
            Book(work_id="w-unknown", title="Uncounted", authors=("D",), ratings_count=None),
        ]
    )
    edges = []
    pairs = [
        ("w-big", "w-mid"),
        ("w-big", "w-tiny"),
        ("w-mid", "w-tiny"),
        ("w-big", "w-unknown"),
    ]
    for a, b in pairs:
        edges.append(Edge(a, b, EdgeKind.GR_SIMILAR, 1, SourceName.DEMO))
        edges.append(Edge(b, a, EdgeKind.GR_SIMILAR, 1, SourceName.DEMO))
    store.insert_raw_edges(edges)
    return store


def _fused_pairs(store) -> set[frozenset[str]]:
    rows = store.conn.execute("SELECT src, dst FROM edges_fused").fetchall()
    return {frozenset(row) for row in rows}


class TestMinRatings:
    def test_no_filter_by_default(self, rated_store) -> None:
        fuse_in_sql(rated_store, GraphConfig())
        assert len(_fused_pairs(rated_store)) == 4

    def test_drops_edges_touching_a_thinly_rated_book(self, rated_store) -> None:
        fuse_in_sql(rated_store, GraphConfig(min_ratings=100))
        pairs = _fused_pairs(rated_store)
        assert frozenset({"w-big", "w-mid"}) in pairs
        # Every edge touching "Obscure" goes, not just the book.
        assert not any("w-tiny" in pair for pair in pairs), pairs

    def test_an_absent_count_is_kept_not_treated_as_zero(self, rated_store) -> None:
        """The defect this test exists for.

        "We do not know how many ratings this has" is not "it has none". Coercing
        a blank to 0 would delete a large slice of the real corpus while looking
        like a working filter.
        """
        fuse_in_sql(rated_store, GraphConfig(min_ratings=100))
        pairs = _fused_pairs(rated_store)
        assert frozenset({"w-big", "w-unknown"}) in pairs, pairs

    def test_a_threshold_above_everything_leaves_no_edges(self, rated_store) -> None:
        fuse_in_sql(rated_store, GraphConfig(min_ratings=10_000_000))
        # Only the unknown-count book survives the filter, and its one edge needs
        # a partner that did not.
        assert _fused_pairs(rated_store) == set()

    def test_zero_is_not_the_same_as_unset(self, rated_store) -> None:
        """``--min-ratings 0`` must be a no-op, not a filter on NULL."""
        fuse_in_sql(rated_store, GraphConfig(min_ratings=0))
        assert len(_fused_pairs(rated_store)) == 4

    def test_weights_are_unchanged_for_surviving_edges(self, rated_store) -> None:
        """The filter selects edges; it must not alter the ones it keeps."""
        fuse_in_sql(rated_store, GraphConfig())
        before = dict(
            rated_store.conn.execute(
                "SELECT src || '|' || dst, weight FROM edges_fused"
            ).fetchall()
        )
        fuse_in_sql(rated_store, GraphConfig(min_ratings=100))
        after = dict(
            rated_store.conn.execute(
                "SELECT src || '|' || dst, weight FROM edges_fused"
            ).fetchall()
        )
        for key, weight in after.items():
            assert before[key] == pytest.approx(weight)


class TestMinRatingsCLI:
    def test_the_flag_is_documented(self) -> None:
        from typer.testing import CliRunner

        from bookmap.cli import app

        result = CliRunner().invoke(app, ["build", "--help"])
        assert result.exit_code == 0
        assert "--min-ratings" in result.output

    def test_it_is_a_no_op_on_a_corpus_with_no_rating_counts(self, tmp_path) -> None:
        """End to end, and the strongest statement of the NULL rule.

        The hand-authored demo corpus carries no ``ratings_count`` at all -- all 177
        books are NULL -- so *no* threshold may remove anything from it. If a future
        change coerced blanks to zero, this build would empty the graph, which is
        exactly the silent failure the filter must not have.

        It also means ``--min-ratings`` does nothing on the demo data. Shrinkage is
        asserted in ``TestMinRatings`` against a store that has real counts.
        """
        import duckdb
        from typer.testing import CliRunner

        from bookmap.cli import app

        runner = CliRunner()
        db = tmp_path / "t.duckdb"
        assert runner.invoke(app, ["ingest", "demo", "--db", str(db)]).exit_code == 0

        def edge_count() -> int:
            conn = duckdb.connect(str(db), read_only=True)
            try:
                return conn.execute("SELECT count(*) FROM edges_fused").fetchone()[0]
            finally:
                conn.close()

        assert runner.invoke(app, ["build", "--db", str(db)]).exit_code == 0
        unfiltered = edge_count()
        assert unfiltered > 0

        assert runner.invoke(
            app, ["build", "--db", str(db), "--min-ratings", "1000000"]
        ).exit_code == 0
        assert edge_count() == unfiltered
