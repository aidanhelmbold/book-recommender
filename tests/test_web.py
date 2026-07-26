"""Web API contract tests.

Everything runs through FastAPI's TestClient -- no server, no browser. The
canvas rendering itself is verified by eye; what is pinned here is the JSON
contract the renderer depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from bookmap.cli import app as cli_app
from bookmap.web.app import create_app


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory) -> Path:
    runner = CliRunner()
    db = tmp_path_factory.mktemp("web") / "demo.duckdb"
    assert runner.invoke(cli_app, ["ingest", "demo", "--db", str(db)]).exit_code == 0
    assert runner.invoke(cli_app, ["build", "--db", str(db)]).exit_code == 0
    return db


@pytest.fixture
def client(demo_db: Path) -> TestClient:
    return TestClient(create_app(str(demo_db)))


class TestIndex:
    def test_serves_the_ui(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_ui_has_a_canvas(self, client: TestClient) -> None:
        assert "<canvas" in client.get("/").text.lower()


class TestSearch:
    def test_finds_a_book(self, client: TestClient) -> None:
        response = client.get("/api/search", params={"q": "dune"})
        assert response.status_code == 200
        results = response.json()["results"]
        assert any("Dune" in r["title"] for r in results)

    def test_result_shape(self, client: TestClient) -> None:
        results = client.get("/api/search", params={"q": "dune"}).json()["results"]
        assert results
        for key in ("work_id", "title", "authors"):
            assert key in results[0]

    def test_empty_query_is_not_an_error(self, client: TestClient) -> None:
        assert client.get("/api/search", params={"q": ""}).status_code == 200

    def test_no_matches_returns_an_empty_list(self, client: TestClient) -> None:
        results = client.get("/api/search", params={"q": "zzzqqxzz"}).json()["results"]
        assert results == []


class TestRecommendEndpoint:
    def test_returns_recommendations(self, client: TestClient) -> None:
        response = client.post("/api/recommend", json={"seeds": ["Dune", "Foundation"], "n": 10})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["recommendations"]) == 10

    def test_recommendation_shape(self, client: TestClient) -> None:
        payload = client.post("/api/recommend", json={"seeds": ["Dune"], "n": 5}).json()
        rec = payload["recommendations"][0]
        for key in ("work_id", "title", "authors", "score", "explanations"):
            assert key in rec

    def test_seeds_are_excluded(self, client: TestClient) -> None:
        payload = client.post("/api/recommend", json={"seeds": ["Dune"], "n": 20}).json()
        assert "Dune" not in [r["title"] for r in payload["recommendations"]]

    def test_unknown_seed_is_reported_not_rejected(self, client: TestClient) -> None:
        """A partially-resolvable seed set is still useful, so this is a 200."""
        response = client.post(
            "/api/recommend", json={"seeds": ["Dune", "Zzzqqx Nonexistent"], "n": 5}
        )
        assert response.status_code == 200
        payload = response.json()
        assert "Zzzqqx Nonexistent" in payload["unresolved"]
        assert payload["recommendations"]

    def test_empty_seed_list_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/recommend", json={"seeds": [], "n": 5}).status_code == 422

    def test_out_of_range_n_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/recommend", json={"seeds": ["Dune"], "n": 0}).status_code == 422
        assert client.post("/api/recommend", json={"seeds": ["Dune"], "n": 9999}).status_code == 422

    def test_beta_changes_the_ranking(self, client: TestClient) -> None:
        """Tuning knobs exposed by the API must actually be wired through."""
        flat = client.post(
            "/api/recommend", json={"seeds": ["Dune"], "n": 15, "beta": 0.0, "mmr_lambda": 1.0}
        ).json()
        damped = client.post(
            "/api/recommend", json={"seeds": ["Dune"], "n": 15, "beta": 0.9, "mmr_lambda": 1.0}
        ).json()
        assert [r["work_id"] for r in flat["recommendations"]] != [
            r["work_id"] for r in damped["recommendations"]
        ]

    def test_malformed_body_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/recommend", json={"nope": True}).status_code == 422


class TestSubgraphEndpoint:
    def test_returns_positioned_nodes_and_edges(self, client: TestClient) -> None:
        response = client.post("/api/subgraph", json={"seeds": ["Dune", "Foundation"], "n": 20})
        assert response.status_code == 200
        payload = response.json()
        assert payload["nodes"]
        assert payload["edges"]

    def test_nodes_carry_layout_and_community(self, client: TestClient) -> None:
        payload = client.post("/api/subgraph", json={"seeds": ["Dune"], "n": 20}).json()
        node = payload["nodes"][0]
        for key in ("work_id", "title", "x", "y", "community", "is_seed"):
            assert key in node

    def test_positions_are_finite(self, client: TestClient) -> None:
        payload = client.post("/api/subgraph", json={"seeds": ["Dune"], "n": 20}).json()
        for node in payload["nodes"]:
            assert isinstance(node["x"], (int, float))
            assert node["x"] == node["x"]  # not NaN
            assert node["y"] == node["y"]

    def test_seeds_are_flagged(self, client: TestClient) -> None:
        payload = client.post("/api/subgraph", json={"seeds": ["Dune"], "n": 20}).json()
        assert any(node["is_seed"] for node in payload["nodes"])

    def test_edges_reference_present_nodes(self, client: TestClient) -> None:
        """A renderer that receives an edge to a missing node draws nothing useful."""
        payload = client.post("/api/subgraph", json={"seeds": ["Dune"], "n": 20}).json()
        ids = {node["work_id"] for node in payload["nodes"]}
        for edge in payload["edges"]:
            assert edge["src"] in ids
            assert edge["dst"] in ids

    def test_community_labels_are_included(self, client: TestClient) -> None:
        payload = client.post("/api/subgraph", json={"seeds": ["Dune"], "n": 20}).json()
        assert "community_labels" in payload


class TestBookEndpoints:
    def _a_work_id(self, client: TestClient) -> str:
        return client.get("/api/search", params={"q": "dune"}).json()["results"][0]["work_id"]

    def test_book_details(self, client: TestClient) -> None:
        work_id = self._a_work_id(client)
        response = client.get(f"/api/book/{work_id}")
        assert response.status_code == 200
        assert response.json()["work_id"] == work_id

    def test_unknown_book_is_404(self, client: TestClient) -> None:
        assert client.get("/api/book/definitely-not-a-work").status_code == 404

    def test_neighbors_for_expansion(self, client: TestClient) -> None:
        work_id = self._a_work_id(client)
        response = client.get(f"/api/book/{work_id}/neighbors")
        assert response.status_code == 200
        neighbors = response.json()["neighbors"]
        assert neighbors
        assert "weight" in neighbors[0]

    def test_neighbors_of_unknown_book_is_404(self, client: TestClient) -> None:
        assert client.get("/api/book/definitely-not-a-work/neighbors").status_code == 404


class TestStatsEndpoint:
    def test_reports_graph_size(self, client: TestClient) -> None:
        payload = client.get("/api/stats").json()
        assert payload["books"] > 0
        assert payload["edges_fused"] > 0


class TestReadOnlySafety:
    def test_serving_never_writes(self, demo_db: Path, tmp_path: Path) -> None:
        """The app must open the database read-only.

        Asserted by behaviour rather than by inspecting the connection: exercise
        every read endpoint and confirm the file is untouched. A write path
        opened by mistake would contend with an ingest, which DuckDB permits only
        one of.
        """
        import shutil

        copy = tmp_path / "copy.duckdb"
        shutil.copy(demo_db, copy)
        before = copy.stat().st_mtime_ns

        client = TestClient(create_app(str(copy)))
        assert client.get("/api/stats").status_code == 200
        client.get("/api/search", params={"q": "dune"})
        client.post("/api/recommend", json={"seeds": ["Dune"], "n": 5})
        client.post("/api/subgraph", json={"seeds": ["Dune"], "n": 10})

        assert copy.stat().st_mtime_ns == before


class TestBridgesEndpoint:
    """Bridges must reach the browser, not just the terminal.

    The map is where "show me how these two genres connect" is most naturally
    seen, so the API half is the one that matters most for this feature.
    """

    def test_returns_bridges(self, client: TestClient) -> None:
        response = client.get("/api/bridges")
        assert response.status_code == 200
        payload = response.json()
        assert payload["bridges"]
        first = payload["bridges"][0]
        for key in ("work_id", "title", "score", "communities", "community_labels"):
            assert key in first
        assert len(first["communities"]) >= 2

    def test_respects_top_n(self, client: TestClient) -> None:
        payload = client.get("/api/bridges", params={"top_n": 3}).json()
        assert len(payload["bridges"]) <= 3

    def test_scores_descend(self, client: TestClient) -> None:
        scores = [b["score"] for b in client.get("/api/bridges").json()["bridges"]]
        assert scores == sorted(scores, reverse=True)

    def test_seeds_restrict_to_relevant_clusters(self, client: TestClient) -> None:
        payload = client.get("/api/bridges", params={"seeds": "Dune,Emma"}).json()
        assert payload["seed_communities"]
        wanted = set(payload["seed_communities"])
        for bridge in payload["bridges"]:
            assert set(bridge["communities"]) & wanted

    def test_unknown_seed_is_reported_not_fatal(self, client: TestClient) -> None:
        response = client.get("/api/bridges", params={"seeds": "Zzzqqx Nonexistent"})
        assert response.status_code == 200
        assert "Zzzqqx Nonexistent" in response.json()["unresolved"]

    def test_out_of_range_top_n_is_rejected(self, client: TestClient) -> None:
        assert client.get("/api/bridges", params={"top_n": 0}).status_code == 422
        assert client.get("/api/bridges", params={"top_n": 9999}).status_code == 422


class TestBridgesInTheUI:
    def test_page_has_a_bridges_panel(self, client: TestClient) -> None:
        """A bare endpoint nobody can see does not surface the feature."""
        body = client.get("/").text.lower()
        assert "bridge" in body

    def test_renderer_marks_bridges_without_a_fourth_colour(self, client: TestClient) -> None:
        """The role palette is capped at three validated hues.

        A fourth would put violet beside blue at deltaE 1.9 under protanopia, so a
        bridge has to be marked by geometry -- a ring or a shape -- rather than by
        adding a colour. This asserts the renderer actually knows about bridges.
        """
        script = client.get("/static/map.js").text
        assert "bridge" in script.lower()


class TestConnectionsEndpoint:
    """The route between the seeds, as data the map can overlay.

    Same payload the CLI prints, because both go through
    ``bookmap.connections.find_connections`` -- two surfaces disagreeing about how
    two books connect would be worse than either being absent.
    """

    def test_returns_a_skeleton(self, client: TestClient) -> None:
        response = client.get("/api/connections", params={"seeds": "Dune,Emma"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["skeletons"], payload
        skeleton = payload["skeletons"][0]
        for key in ("terminals", "connectors", "nodes", "edges", "legs", "cost"):
            assert key in skeleton

    def test_legs_carry_titles_alongside_ids(self, client: TestClient) -> None:
        """The client needs both: ids to re-query with, titles to display."""
        leg = client.get(
            "/api/connections", params={"seeds": "Dune,Emma"}
        ).json()["skeletons"][0]["legs"][0]
        assert len(leg["path"]) == len(leg["titles"])
        assert all(title for title in leg["titles"])
        assert leg["hops"] == len(leg["path"]) - 1

    def test_edges_reference_nodes_on_the_skeleton(self, client: TestClient) -> None:
        """The renderer draws these directly; an edge to a node it was never sent
        would be dropped silently and the route would look broken."""
        skeleton = client.get(
            "/api/connections", params={"seeds": "Dune,Emma"}
        ).json()["skeletons"][0]
        ids = {node["work_id"] for node in skeleton["nodes"]}
        for edge in skeleton["edges"]:
            assert edge["source"] in ids
            assert edge["target"] in ids

    def test_connectors_are_named_and_are_not_seeds(self, client: TestClient) -> None:
        skeleton = client.get(
            "/api/connections", params={"seeds": "Dune,Emma"}
        ).json()["skeletons"][0]
        terminals = {node["work_id"] for node in skeleton["terminals"]}
        for connector in skeleton["connectors"]:
            assert connector["title"]
            assert connector["work_id"] not in terminals

    def test_a_single_seed_is_not_an_error(self, client: TestClient) -> None:
        response = client.get("/api/connections", params={"seeds": "Dune"})
        assert response.status_code == 200
        assert response.json()["skeletons"] == []

    def test_unknown_seed_is_reported_not_fatal(self, client: TestClient) -> None:
        response = client.get(
            "/api/connections", params={"seeds": "Dune,Emma,Zzzqqx Nonexistent"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert "Zzzqqx Nonexistent" in payload["unresolved"]
        assert payload["skeletons"], "the resolvable seeds should still be joined"

    def test_no_seeds_is_rejected(self, client: TestClient) -> None:
        assert client.get("/api/connections", params={"seeds": " , "}).status_code == 422

    def test_hop_cap_is_reported_not_hidden(self, client: TestClient) -> None:
        payload = client.get(
            "/api/connections", params={"seeds": "Dune,Emma", "max_hops": 1}
        ).json()
        assert payload["capped"], payload
        pair = payload["capped"][0]
        assert pair["limit"] == 1
        assert pair["hops"] > 1
        assert pair["source_title"] and pair["target_title"]

    def test_out_of_range_max_hops_is_rejected(self, client: TestClient) -> None:
        for value in (0, 999):
            response = client.get(
                "/api/connections", params={"seeds": "Dune,Emma", "max_hops": value}
            )
            assert response.status_code == 422, value


class TestConnectionsInTheUI:
    def test_the_page_offers_a_connections_view(self, client: TestClient) -> None:
        """An endpoint nobody can reach does not surface the feature."""
        body = client.get("/").text.lower()
        assert "connection" in body

    def test_the_renderer_marks_connectors_by_shape(self, client: TestClient) -> None:
        """Not by a fourth hue.

        The role palette is capped at three validated slots; a ring is taken by
        seeds and a double ring by bridges, so a connector gets the remaining
        geometric slot. Colour must not be the carrier.
        """
        script = client.get("/static/map.js").text.lower()
        assert "connector" in script
        assert "diamond" in script


class TestConnectionsOnTheSubgraph:
    """The skeleton has to reach the *layout*, not just the sidebar.

    This is the whole point of the feature: connector books are excluded from the
    neighbourhood by construction -- a book joining SF to Regency romance is not
    among the most similar to either side -- so unless the subgraph explicitly
    carries them, the lobes stay apart no matter what the force layout does. And
    once the cross-lobe edges are really there, the layout arranges the clusters
    along the connecting spine by itself.
    """

    def test_connector_nodes_are_on_the_map(self, client: TestClient) -> None:
        route = client.get("/api/connections", params={"seeds": "Dune,Emma"}).json()
        connectors = {
            node["work_id"]
            for skeleton in route["skeletons"]
            for node in skeleton["connectors"]
        }
        assert connectors, "no connectors to check"
        payload = client.post(
            "/api/subgraph", json={"seeds": ["Dune", "Emma"], "n": 30}
        ).json()
        drawn = {node["work_id"] for node in payload["nodes"]}
        assert connectors <= drawn, connectors - drawn

    def test_the_payload_names_its_connectors(self, client: TestClient) -> None:
        """The renderer marks connectors by shape, so it must know which they are."""
        payload = client.post(
            "/api/subgraph", json={"seeds": ["Dune", "Emma"], "n": 30}
        ).json()
        assert payload["connectors"]
        drawn = {node["work_id"] for node in payload["nodes"]}
        assert set(payload["connectors"]) <= drawn

    def test_skeleton_edges_are_drawable(self, client: TestClient) -> None:
        payload = client.post(
            "/api/subgraph", json={"seeds": ["Dune", "Emma"], "n": 30}
        ).json()
        assert payload["skeleton_edges"]
        drawn = {node["work_id"] for node in payload["nodes"]}
        for edge in payload["skeleton_edges"]:
            assert edge["src"] in drawn
            assert edge["dst"] in drawn

    def test_routes_carry_a_quotable_sentence(self, client: TestClient) -> None:
        payload = client.post(
            "/api/subgraph", json={"seeds": ["Dune", "Emma"], "n": 30}
        ).json()
        assert payload["routes"]
        route = payload["routes"][0]
        assert len(route["titles"]) >= 2
        assert route["hops"] == len(route["titles"]) - 1

    def test_connectors_keep_the_three_hue_palette(self, client: TestClient) -> None:
        """A connector must not introduce a fourth role colour.

        Only three categorical hues clear the colour-vision floors for an
        all-pairs form, so a connector's identity rides on geometry and its fill
        stays one of the three existing roles.
        """
        payload = client.post(
            "/api/subgraph", json={"seeds": ["Dune", "Emma"], "n": 30}
        ).json()
        roles = {node["role"] for node in payload["nodes"]}
        assert roles <= {"seed", "recommendation", "context"}

    def test_connections_can_be_switched_off(self, client: TestClient) -> None:
        """The neighbourhood view must stay exactly what it was.

        The skeleton is an overlay; a caller asking for the plain neighbourhood
        must not silently get extra nodes in it.
        """
        payload = client.post(
            "/api/subgraph",
            json={"seeds": ["Dune", "Emma"], "n": 30, "include_connections": False},
        ).json()
        assert payload["connectors"] == []
        assert payload["skeleton_edges"] == []

    def test_a_single_seed_still_renders(self, client: TestClient) -> None:
        payload = client.post("/api/subgraph", json={"seeds": ["Dune"], "n": 20}).json()
        assert payload["nodes"]
        assert payload["connectors"] == []


class TestConnectionsObjectiveParams:
    """The route objective is a real question on real data, so the API exposes it.

    The default stays `product` because which objective reads better is still open
    — but the endpoint must not lock a caller out of the alternative.
    """

    def test_widest_is_accepted(self, client: TestClient) -> None:
        response = client.get(
            "/api/connections", params={"seeds": "Dune,Emma", "objective": "widest"}
        )
        assert response.status_code == 200
        assert response.json()["objective"] == "widest"

    def test_an_unknown_objective_is_rejected(self, client: TestClient) -> None:
        response = client.get(
            "/api/connections", params={"seeds": "Dune,Emma", "objective": "cheapest-ish"}
        )
        assert response.status_code == 422

    def test_a_weight_floor_is_accepted_and_echoed(self, client: TestClient) -> None:
        payload = client.get(
            "/api/connections",
            params={"seeds": "Dune,Emma", "min_edge_weight": 0.15},
        ).json()
        assert payload["min_edge_weight"] == pytest.approx(0.15)
        for skeleton in payload["skeletons"]:
            for leg in skeleton["legs"]:
                assert leg["bottleneck"] >= 0.15

    def test_an_out_of_range_floor_is_rejected(self, client: TestClient) -> None:
        for value in (-0.1, 1.5):
            response = client.get(
                "/api/connections", params={"seeds": "Dune,Emma", "min_edge_weight": value}
            )
            assert response.status_code == 422, value

    def test_the_default_is_unchanged(self, client: TestClient) -> None:
        payload = client.get("/api/connections", params={"seeds": "Dune,Emma"}).json()
        assert payload["objective"] == "product"
        assert payload["min_edge_weight"] == 0.0
