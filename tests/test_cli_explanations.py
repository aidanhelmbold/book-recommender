"""Explanation rendering must name books, not internal ids.

An explanation exists to be read and sanity-checked by a human: "2 hops from
Dune via Dune Messiah" is checkable, "2 hops from Dune via the-dispossessed"
leaks a work id and reads like a bug even when the path is correct. The waypoints
in the middle of a path are exactly the nodes that are neither a seed nor a
recommendation, so they are the ones a naive titles map misses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bookmap.cli import explanation_titles
from bookmap.graph.build import fuse_edges
from bookmap.recommend import recommend
from bookmap.sources.demo import DemoSource
from bookmap.store.db import Store
from bookmap.store.projection import project


@pytest.fixture(scope="module")
def demo_store_path(tmp_path_factory) -> Path:
    db = tmp_path_factory.mktemp("cli_explain") / "demo.duckdb"
    source = DemoSource()
    with Store.open(db) as store:
        store.upsert_books(source.iter_books())
        store.insert_raw_edges(source.iter_edges())
        store.resolve_refs()
        store.replace_fused_edges(fuse_edges(store.iter_raw_edges()))
    return db


def test_every_path_node_resolves_to_a_real_title(demo_store_path: Path) -> None:
    with Store.open(demo_store_path, read_only=True) as store:
        projection = project(store)
        result = recommend(["Dune", "Foundation", "Hyperion"], store, projection, n=10)
        titles = explanation_titles(result, store)

        assert result.recommendations
        seen_waypoint = False
        for rec in result.recommendations:
            for explanation in rec.explanations:
                for work_id in explanation.path:
                    assert work_id in titles, f"{work_id} missing from titles map"
                    # The demo corpus keys books by lowercase slug, so a title
                    # that still equals its own id was never looked up.
                    assert titles[work_id] != work_id, (
                        f"{work_id} rendered as its work id rather than a title"
                    )
                for work_id in explanation.path[1:-1]:
                    seen_waypoint = True
                    assert titles[work_id] != work_id

        assert seen_waypoint, "no multi-hop explanation in the sample; test proves nothing"


def test_cli_renders_waypoint_titles_end_to_end(tmp_path: Path) -> None:
    """Drive the real command, because the unit tests above cannot catch this.

    Naming a waypoint requires a store lookup, and that lookup only happens when
    a path walks through a book that is neither a seed nor a result. A rendering
    path that touched an already-closed store therefore passed every existing
    test while crashing on the first multi-hop explanation a user saw. This
    asserts on the rendered text with a seed set known to produce one.
    """
    from typer.testing import CliRunner

    from bookmap.cli import app

    runner = CliRunner()
    db = tmp_path / "demo.duckdb"
    assert runner.invoke(app, ["ingest", "demo", "--db", str(db)]).exit_code == 0
    assert runner.invoke(app, ["build", "--db", str(db)]).exit_code == 0

    result = runner.invoke(
        app, ["recommend", "--seeds", "Dune,Hyperion", "-n", "8", "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert "via" in result.output, "no multi-hop explanation rendered"

    for line in result.output.splitlines():
        _, separator, tail = line.partition(" via ")
        if not separator:
            continue
        for waypoint in tail.split(","):
            name = waypoint.strip()
            # Demo work ids are lowercase slugs while titles are capitalised, so
            # an all-lowercase waypoint is an unresolved id leaking through.
            assert name and name != name.lower(), f"work id leaked into output: {name!r}"


def test_titles_map_covers_seeds_and_recommendations(demo_store_path: Path) -> None:
    with Store.open(demo_store_path, read_only=True) as store:
        projection = project(store)
        result = recommend(["Dune", "Emma"], store, projection, n=5)
        titles = explanation_titles(result, store)
        for match in result.seeds:
            assert titles[match.work_id] == match.title
        for rec in result.recommendations:
            assert titles[rec.work_id] == rec.title
