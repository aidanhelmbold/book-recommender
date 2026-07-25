"""CLI surface tests.

These run the real commands against a temp database, so they double as the
end-to-end check that ingest, build, and recommend actually compose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookmap.cli import app

runner = CliRunner()


@pytest.fixture
def demo_db(tmp_path: Path) -> Path:
    """A database built from the bundled demo corpus via the CLI itself."""
    db = tmp_path / "demo.duckdb"
    result = runner.invoke(app, ["ingest", "demo", "--db", str(db)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["build", "--db", str(db)])
    assert result.exit_code == 0, result.output
    return db


class TestHelp:
    def test_root_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("ingest", "build", "recommend", "map", "web", "stats"):
            assert command in result.output

    def test_ingest_help_lists_sources(self) -> None:
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "goodreads-ucsd" in result.output
        assert "amazon-meta" in result.output


class TestIngest:
    def test_demo_populates_the_store(self, tmp_path: Path) -> None:
        db = tmp_path / "t.duckdb"
        result = runner.invoke(app, ["ingest", "demo", "--db", str(db)])
        assert result.exit_code == 0, result.output
        from bookmap.store.db import Store

        with Store.open(db, read_only=True) as store:
            assert store.counts()["books"] >= 150
            assert store.counts()["edges_raw"] >= 500

    def test_missing_dump_file_fails_cleanly(self, tmp_path: Path) -> None:
        """A wrong path should produce a readable error, not a traceback."""
        result = runner.invoke(
            app,
            ["ingest", "goodreads-ucsd", str(tmp_path / "nope.json.gz"), "--db", str(tmp_path / "t.duckdb")],
        )
        assert result.exit_code != 0
        assert "nope.json.gz" in result.output or "not found" in result.output.lower()


class TestBuild:
    def test_creates_the_fused_graph(self, demo_db: Path) -> None:
        from bookmap.store.db import Store

        with Store.open(demo_db, read_only=True) as store:
            counts = store.counts()
            assert counts["edges_fused"] > 0
            assert counts["node_metrics"] > 0

    def test_assigns_communities(self, demo_db: Path) -> None:
        from bookmap.store.db import Store

        with Store.open(demo_db, read_only=True) as store:
            assert store.counts()["communities"] >= 2

    def test_is_rerunnable(self, demo_db: Path) -> None:
        """Rebuilding must replace the graph, not compound it."""
        from bookmap.store.db import Store

        with Store.open(demo_db, read_only=True) as store:
            before = store.counts()["edges_fused"]
        result = runner.invoke(app, ["build", "--db", str(demo_db)])
        assert result.exit_code == 0, result.output
        with Store.open(demo_db, read_only=True) as store:
            assert store.counts()["edges_fused"] == before


class TestRecommend:
    def test_returns_results(self, demo_db: Path) -> None:
        result = runner.invoke(
            app, ["recommend", "--seeds", "Dune,Foundation", "-n", "5", "--db", str(demo_db)]
        )
        assert result.exit_code == 0, result.output
        assert result.output.strip()

    def test_json_output_is_parseable(self, demo_db: Path) -> None:
        result = runner.invoke(
            app,
            ["recommend", "--seeds", "Dune", "-n", "5", "--json", "--db", str(demo_db)],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload["recommendations"]) == 5
        assert payload["recommendations"][0]["title"]

    def test_seeds_are_absent_from_output(self, demo_db: Path) -> None:
        result = runner.invoke(
            app, ["recommend", "--seeds", "Dune", "-n", "10", "--json", "--db", str(demo_db)]
        )
        payload = json.loads(result.output)
        assert "Dune" not in [r["title"] for r in payload["recommendations"]]

    def test_explanations_are_shown_by_default(self, demo_db: Path) -> None:
        result = runner.invoke(
            app, ["recommend", "--seeds", "Dune", "-n", "3", "--db", str(demo_db)]
        )
        assert result.exit_code == 0, result.output
        assert "Dune" in result.output  # cited as the reason

    def test_unknown_seed_is_reported(self, demo_db: Path) -> None:
        result = runner.invoke(
            app,
            ["recommend", "--seeds", "Zzzqqx Nonexistent", "-n", "5", "--db", str(demo_db)],
        )
        assert "Zzzqqx" in result.output

    def test_missing_database_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["recommend", "--seeds", "Dune", "--db", str(tmp_path / "absent.duckdb")]
        )
        assert result.exit_code != 0


class TestMap:
    def test_writes_a_standalone_html_file(self, demo_db: Path, tmp_path: Path) -> None:
        out = tmp_path / "map.html"
        result = runner.invoke(
            app,
            ["map", "--seeds", "Dune,Foundation", "--out", str(out), "--db", str(demo_db)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "<canvas" in html.lower() or "<svg" in html.lower()

    def test_html_is_self_contained(self, demo_db: Path, tmp_path: Path) -> None:
        """The export must open from disk with no network access."""
        out = tmp_path / "map.html"
        runner.invoke(
            app, ["map", "--seeds", "Dune", "--out", str(out), "--db", str(demo_db)]
        )
        html = out.read_text(encoding="utf-8")
        assert "src=\"http" not in html
        assert "href=\"http" not in html.replace('href="https://www.goodreads', "")


class TestStats:
    def test_prints_graph_size(self, demo_db: Path) -> None:
        result = runner.invoke(app, ["stats", "--db", str(demo_db)])
        assert result.exit_code == 0, result.output
        assert "books" in result.output.lower()
