"""``bookmap route-report`` exists to make a judgement call cheap to gather.

Whether a route reads as *meaningful* rather than merely *short* cannot be
asserted in a test — it needs someone who knows the books. What a test can do is
guarantee the report puts everything that judgement needs in one place: which book
each seed actually resolved to, the route under each objective, and the weakest
link on it.

The seed-resolution line is not decoration. "Atomic Habits" silently resolving to
"Atomic: An I Bring the Fire Short Story" produced an evaluation of the wrong
books, and without the resolved titles on the page there was no way to see it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookmap.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def built_db(tmp_path_factory) -> Path:
    db = tmp_path_factory.mktemp("report") / "demo.duckdb"
    assert runner.invoke(app, ["ingest", "demo", "--db", str(db)]).exit_code == 0
    assert runner.invoke(app, ["build", "--db", str(db)]).exit_code == 0
    return db


class TestRouteReport:
    def test_runs_with_built_in_probes(self, built_db: Path) -> None:
        result = runner.invoke(app, ["route-report", "--db", str(built_db)])
        assert result.exit_code == 0, result.output
        assert "# Route comparison" in result.output

    def test_covers_every_objective_for_each_set(self, built_db: Path) -> None:
        result = runner.invoke(
            app, ["route-report", "--db", str(built_db), "--sets", "Dune,Emma"]
        )
        assert result.exit_code == 0, result.output
        for label in ("product", "product + floor", "widest"):
            assert label in result.output, label

    def test_names_what_each_seed_resolved_to(self, built_db: Path) -> None:
        """The check that catches an evaluation of the wrong books."""
        result = runner.invoke(
            app, ["route-report", "--db", str(built_db), "--sets", "Dune,Emma"]
        )
        assert "resolved" in result.output.lower()
        assert "Frank Herbert" in result.output

    def test_reports_the_weakest_link_and_timing(self, built_db: Path) -> None:
        result = runner.invoke(
            app, ["route-report", "--db", str(built_db), "--sets", "Dune,Emma"]
        )
        lowered = result.output.lower()
        assert "weakest" in lowered
        assert "seconds" in lowered or "ms" in lowered

    def test_states_the_graph_size(self, built_db: Path) -> None:
        """A report without the graph it ran against cannot be compared to another."""
        result = runner.invoke(
            app, ["route-report", "--db", str(built_db), "--sets", "Dune,Emma"]
        )
        assert "nodes" in result.output and "edges" in result.output

    def test_multiple_sets_are_each_reported(self, built_db: Path) -> None:
        result = runner.invoke(
            app,
            ["route-report", "--db", str(built_db),
             "--sets", "Dune,Emma", "--sets", "Dune,Meditations"],
        )
        assert result.exit_code == 0, result.output
        assert result.output.count("seeds resolved") >= 2

    def test_writes_a_file_when_asked(self, built_db: Path, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        result = runner.invoke(
            app,
            ["route-report", "--db", str(built_db), "--sets", "Dune,Emma", "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        body = out.read_text(encoding="utf-8")
        assert "# Route comparison" in body
        assert "widest" in body

    def test_too_few_usable_seeds_says_why(self, built_db: Path) -> None:
        """Not "no route found" -- that would blame the graph for a seed problem.

        Repeating a graph-shaped answer once per objective would also imply three
        routings happened when none could.
        """
        result = runner.invoke(
            app,
            ["route-report", "--db", str(built_db), "--sets", "Dune,Zzzqx Not A Book"],
        )
        assert result.exit_code == 0, result.output
        assert "no route found" not in result.output.lower()
        assert "fewer than two" in result.output.lower()

    def test_an_unresolvable_set_is_reported_not_fatal(self, built_db: Path) -> None:
        """One bad probe must not abandon the rest of the battery."""
        result = runner.invoke(
            app,
            ["route-report", "--db", str(built_db),
             "--sets", "Zzzqx Not A Book,Qqqzx Also Not", "--sets", "Dune,Emma"],
        )
        assert result.exit_code == 0, result.output
        assert "Zzzqx" in result.output
        assert "Frank Herbert" in result.output

    def test_a_single_seed_set_is_rejected(self, built_db: Path) -> None:
        """A route needs two ends; a one-book probe is a typo, not a query."""
        result = runner.invoke(
            app, ["route-report", "--db", str(built_db), "--sets", "Dune"]
        )
        assert result.exit_code != 0

    def test_missing_database_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["route-report", "--db", str(tmp_path / "nope.duckdb")]
        )
        assert result.exit_code != 0
