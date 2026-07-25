"""The CLI must expose DuckDB's spill controls.

This exists because of a concrete failure: ingesting a 9.2 GB dump filled the
user's disk. DuckDB's default temp directory sits beside the database file and
its default size cap is a fraction of the whole *volume*, so a query that spills
consumes the disk and takes the machine with it. Naming a directory and a
ceiling turns that into a query that fails quickly, saying which limit it hit --
but only if the write-side commands actually accept the options.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from bookmap.cli import app

runner = CliRunner()


class TestResourceOptions:
    def test_ingest_accepts_spill_controls(self, tmp_path: Path) -> None:
        spill = tmp_path / "spill"
        result = runner.invoke(
            app,
            [
                "ingest", "demo",
                "--db", str(tmp_path / "t.duckdb"),
                "--temp-dir", str(spill),
                "--memory-limit", "2GB",
                "--max-temp-size", "4GB",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_build_accepts_spill_controls(self, tmp_path: Path) -> None:
        db = tmp_path / "t.duckdb"
        assert runner.invoke(app, ["ingest", "demo", "--db", str(db)]).exit_code == 0
        result = runner.invoke(
            app,
            ["build", "--db", str(db), "--temp-dir", str(tmp_path / "spill"),
             "--memory-limit", "2GB", "--max-temp-size", "4GB"],
        )
        assert result.exit_code == 0, result.output

    def test_options_are_documented(self) -> None:
        """A knob nobody can find does not solve the problem it was added for."""
        for command in (["ingest", "goodreads-ucsd", "--help"], ["build", "--help"]):
            result = runner.invoke(app, command)
            assert result.exit_code == 0
            assert "--temp-dir" in result.output
            assert "--memory-limit" in result.output

    def test_a_too_small_memory_limit_still_completes(self, tmp_path: Path) -> None:
        """Spilling is expected under a tight limit, not fatal.

        The point of the ceiling is that spill stays bounded and observable; a
        small limit should push work to disk and finish, not error out.
        """
        result = runner.invoke(
            app,
            ["ingest", "demo", "--db", str(tmp_path / "t.duckdb"),
             "--memory-limit", "256MB", "--temp-dir", str(tmp_path / "spill")],
        )
        assert result.exit_code == 0, result.output

    def test_bad_memory_limit_fails_cleanly(self, tmp_path: Path) -> None:
        """A malformed limit should be a readable error, not a traceback."""
        result = runner.invoke(
            app,
            ["ingest", "demo", "--db", str(tmp_path / "t.duckdb"),
             "--memory-limit", "not-a-size"],
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output


class TestMissingDatabaseMessage:
    """A wrong --db is the likeliest mistake once more than one database exists.

    The original message only suggested `ingest demo`, which is actively
    misleading for someone who has a real database at another path -- the fix is
    to name the flag they forgot.
    """

    def test_names_the_db_flag(self, tmp_path: Path) -> None:
        for command in (["web"], ["stats"], ["recommend", "--seeds", "Dune"]):
            result = runner.invoke(app, [*command, "--db", str(tmp_path / "absent.duckdb")])
            assert result.exit_code != 0
            assert "--db" in result.output, f"{command} did not mention --db"
            assert "absent.duckdb" in result.output
