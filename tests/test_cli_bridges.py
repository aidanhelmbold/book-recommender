"""Bridge books must be reachable, not merely computed.

``graph/centrality.py`` has found these since Wave 1 and nothing has ever shown
them. A bridge book is a title carrying an unusual share of its edge weight
*across* community boundaries — the book that links two otherwise-separate reading
communities. For someone trying to move from one genre into another that is the
most useful single recommendation the graph can make, and plain similarity ranking
cannot surface it: a bridge is by definition not the most similar thing to either
side.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bookmap.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def built_db(tmp_path_factory) -> Path:
    db = tmp_path_factory.mktemp("bridges") / "demo.duckdb"
    assert runner.invoke(app, ["ingest", "demo", "--db", str(db)]).exit_code == 0
    assert runner.invoke(app, ["build", "--db", str(db)]).exit_code == 0
    return db


class TestBridgesCommand:
    def test_lists_bridge_books(self, built_db: Path) -> None:
        result = runner.invoke(app, ["bridges", "--db", str(built_db)])
        assert result.exit_code == 0, result.output
        assert result.output.strip()

    def test_json_output_is_parseable_and_shaped(self, built_db: Path) -> None:
        result = runner.invoke(app, ["bridges", "--db", str(built_db), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["bridges"]
        first = payload["bridges"][0]
        for key in ("work_id", "title", "score", "communities"):
            assert key in first
        # A bridge that touches only one community is not a bridge.
        assert len(first["communities"]) >= 2

    def test_names_the_communities_it_connects(self, built_db: Path) -> None:
        """A bare score is useless; the point is *which* two worlds it joins."""
        result = runner.invoke(app, ["bridges", "--db", str(built_db), "--json"])
        payload = json.loads(result.output)
        labelled = [b for b in payload["bridges"] if any(b["community_labels"])]
        assert labelled, "no bridge reported a human-readable community label"

    def test_respects_top_n(self, built_db: Path) -> None:
        result = runner.invoke(app, ["bridges", "--db", str(built_db), "--top-n", "3", "--json"])
        assert len(json.loads(result.output)["bridges"]) <= 3

    def test_scores_are_descending(self, built_db: Path) -> None:
        result = runner.invoke(app, ["bridges", "--db", str(built_db), "--json"])
        scores = [b["score"] for b in json.loads(result.output)["bridges"]]
        assert scores == sorted(scores, reverse=True)

    def test_seeds_restrict_to_relevant_communities(self, built_db: Path) -> None:
        """With seeds given, only bridges touching the seeds' clusters are useful.

        Someone who reads hard SF and Regency romance wants the books joining
        *those* two worlds, not the strongest bridge somewhere else in the graph.
        """
        result = runner.invoke(
            app, ["bridges", "--db", str(built_db), "--seeds", "Dune,Emma", "--json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["seed_communities"], "seed communities not reported"
        seed_communities = set(payload["seed_communities"])
        for bridge in payload["bridges"]:
            assert set(bridge["communities"]) & seed_communities, (
                f"{bridge['title']} touches none of the seeded clusters"
            )

    def test_unknown_seed_is_reported_not_fatal(self, built_db: Path) -> None:
        result = runner.invoke(
            app, ["bridges", "--db", str(built_db), "--seeds", "Zzzqqx Nonexistent", "--json"]
        )
        assert result.exit_code == 0, result.output
        assert "Zzzqqx Nonexistent" in json.loads(result.output)["unresolved"]

    def test_missing_database_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["bridges", "--db", str(tmp_path / "absent.duckdb")])
        assert result.exit_code != 0
        assert "--db" in result.output

    def test_ungrouped_graph_says_so(self, tmp_path: Path) -> None:
        """Without communities there is nothing to bridge between.

        Ingesting without building leaves node_metrics empty, and the honest
        response is to say which command is missing rather than report zero
        bridges as though that were a finding.
        """
        db = tmp_path / "raw.duckdb"
        assert runner.invoke(app, ["ingest", "demo", "--db", str(db)]).exit_code == 0
        result = runner.invoke(app, ["bridges", "--db", str(db)])
        assert result.exit_code != 0
        assert "build" in result.output.lower()

    def test_documented_in_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "bridges" in result.output
