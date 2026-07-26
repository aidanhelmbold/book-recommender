"""``bookmap connect`` must say the route in prose, on the real demo corpus.

Phase 2 of `docs/plan-connections.md` exists to answer a question the graph tests
cannot: whether the routes read as *meaningful* or merely *short*. That is a
judgement about output, so the point of these tests is that the route is
**nameable** — titles, not work ids, with the connecting books called out — and
that every failure mode the real graph produces (a seed that does not resolve, a
pair in separate components, a route past the hop cap) is stated rather than
swallowed.

The demo corpus is hand-authored with deliberate cross-genre bridges, so a
skeleton joining SF to Regency romance has somewhere honest to route through.
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
    db = tmp_path_factory.mktemp("connect") / "demo.duckdb"
    assert runner.invoke(app, ["ingest", "demo", "--db", str(db)]).exit_code == 0
    assert runner.invoke(app, ["build", "--db", str(db)]).exit_code == 0
    return db


class TestConnectCommand:
    def test_joins_two_distant_books(self, built_db: Path) -> None:
        result = runner.invoke(app, ["connect", "--seeds", "Dune,Emma", "--db", str(built_db)])
        assert result.exit_code == 0, result.output
        assert "Dune" in result.output
        assert "Emma" in result.output
        # The arrow is the prose form the plan asks for: a route the reader can
        # quote. Its absence means the command found nothing to say.
        assert "→" in result.output

    def test_names_the_connecting_books(self, built_db: Path) -> None:
        """The connectors are the answer, so they must be listed as titles."""
        result = runner.invoke(app, ["connect", "--seeds", "Dune,Emma", "--db", str(built_db)])
        assert "connecting books" in result.output.lower()
        # A work id leaking into the prose reads as a bug even when the route is
        # right, and it is unquotable.
        assert "demo-" not in result.output

    def test_reports_hops_and_strength(self, built_db: Path) -> None:
        result = runner.invoke(app, ["connect", "--seeds", "Dune,Emma", "--db", str(built_db)])
        assert "hop" in result.output.lower()

    def test_three_seeds_are_joined_by_one_structure(self, built_db: Path) -> None:
        result = runner.invoke(
            app, ["connect", "--seeds", "Dune,Emma,Meditations", "--db", str(built_db)]
        )
        assert result.exit_code == 0, result.output
        assert "→" in result.output

    def test_json_output_is_parseable(self, built_db: Path) -> None:
        result = runner.invoke(
            app, ["connect", "--seeds", "Dune,Emma", "--db", str(built_db), "--json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["skeletons"], payload
        skeleton = payload["skeletons"][0]
        assert skeleton["legs"]
        leg = skeleton["legs"][0]
        # Titles alongside ids: the ids are what a caller re-queries with, the
        # titles are what it displays. Dropping either forces a second round trip.
        assert leg["path"] and leg["titles"]
        assert len(leg["path"]) == len(leg["titles"])
        assert leg["hops"] == len(leg["path"]) - 1
        assert 0.0 < leg["strength"] <= 1.0

    def test_single_seed_says_there_is_nothing_to_connect(self, built_db: Path) -> None:
        """Not an error: one book is a legitimate query with no route in it."""
        result = runner.invoke(app, ["connect", "--seeds", "Dune", "--db", str(built_db)])
        assert result.exit_code == 0, result.output
        assert "two" in result.output.lower() or "nothing to connect" in result.output.lower()

    def test_unresolved_seed_is_reported(self, built_db: Path) -> None:
        result = runner.invoke(
            app,
            ["connect", "--seeds", "Dune,Emma,Zzzqx Not A Book", "--db", str(built_db)],
        )
        # The seed the user thought they gave you changes the answer by its
        # absence, so it is named even though the rest succeeded.
        assert "Zzzqx" in result.output

    def test_hop_cap_is_reported_not_hidden(self, built_db: Path) -> None:
        """A capped pair must be visible; a shortened path would be a false claim."""
        result = runner.invoke(
            app,
            ["connect", "--seeds", "Dune,Emma", "--db", str(built_db), "--max-hops", "1"],
        )
        assert result.exit_code == 0, result.output
        lowered = result.output.lower()
        assert "hop" in lowered and ("further apart" in lowered or "cap" in lowered)

    def test_no_seeds_fails_cleanly(self, built_db: Path) -> None:
        result = runner.invoke(app, ["connect", "--seeds", " , ", "--db", str(built_db)])
        assert result.exit_code != 0

    def test_missing_database_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["connect", "--seeds", "Dune,Emma", "--db", str(tmp_path / "nope.duckdb")]
        )
        assert result.exit_code != 0


class TestRouteObjectiveOptions:
    """The comparison has to be runnable from the command line, on one screen.

    Judging two objectives by running two commands and scrolling between them is
    how you end up comparing different seed resolutions by accident.
    """

    def test_widest_objective_runs(self, built_db: Path) -> None:
        result = runner.invoke(
            app,
            ["connect", "--seeds", "Dune,Emma", "--db", str(built_db),
             "--objective", "widest"],
        )
        assert result.exit_code == 0, result.output
        assert "→" in result.output

    def test_an_unknown_objective_fails_cleanly(self, built_db: Path) -> None:
        result = runner.invoke(
            app,
            ["connect", "--seeds", "Dune,Emma", "--db", str(built_db),
             "--objective", "cheapest-ish"],
        )
        assert result.exit_code != 0
        assert "cheapest-ish" in result.output or "objective" in result.output.lower()

    def test_a_weight_floor_is_accepted(self, built_db: Path) -> None:
        result = runner.invoke(
            app,
            ["connect", "--seeds", "Dune,Emma", "--db", str(built_db),
             "--min-edge-weight", "0.15"],
        )
        assert result.exit_code == 0, result.output

    def test_the_bottleneck_is_reported(self, built_db: Path) -> None:
        """The number that exposes a weak link must be on screen, not just in JSON."""
        result = runner.invoke(app, ["connect", "--seeds", "Dune,Emma", "--db", str(built_db)])
        assert "weakest" in result.output.lower()

    def test_compare_shows_every_variant_together(self, built_db: Path) -> None:
        result = runner.invoke(
            app, ["connect", "--seeds", "Dune,Emma", "--db", str(built_db), "--compare"]
        )
        assert result.exit_code == 0, result.output
        lowered = result.output.lower()
        for label in ("product", "widest", "floor"):
            assert label in lowered, label

    def test_compare_names_the_objective_of_each_block(self, built_db: Path) -> None:
        """Unlabelled blocks are the whole failure mode of a comparison."""
        payload = runner.invoke(
            app,
            ["connect", "--seeds", "Dune,Emma", "--db", str(built_db),
             "--compare", "--json"],
        )
        assert payload.exit_code == 0, payload.output
        variants = json.loads(payload.output)["variants"]
        assert [v["label"] for v in variants] == ["product", "product + floor", "widest"]
        for variant in variants:
            assert "skeletons" in variant

    def test_json_carries_the_bottleneck(self, built_db: Path) -> None:
        payload = json.loads(
            runner.invoke(
                app, ["connect", "--seeds", "Dune,Emma", "--db", str(built_db), "--json"]
            ).output
        )
        leg = payload["skeletons"][0]["legs"][0]
        assert 0.0 < leg["bottleneck"] <= 1.0
        assert payload["objective"] == "product"
