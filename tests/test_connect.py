"""The Steiner skeleton: the minimal structure joining a set of seeds.

Seeding three unrelated books draws three disconnected lobes, because the
subgraph is assembled by *relevance* — and a book that joins SF to Regency
romance is by definition not among the most similar to either side. This module
finds the books that do the joining.

Everything here is checked against an independent oracle rather than a snapshot:
NetworkX's ``shortest_path_length`` for the metric closure, its
``approximation.steiner_tree`` for the skeleton itself, and a *planted* bridge —
two cliques joined by exactly one node — for the cases where both must agree.
The last of those is the strongest evidence available: ``bridge_books`` scores by
cross-community edge weight and the skeleton is found by shortest path, so they
are genuinely independent methods.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from bookmap.explain import edge_cost
from bookmap.graph.connect import (
    DEFAULT_MAX_HOPS,
    Connections,
    connect_seeds,
    cost_matrix,
)
from tests.helpers import make_projection


def _nx_with_costs(pairs: list[tuple[str, str, float]]) -> nx.Graph:
    """The same graph as ``make_projection``, with ``-log(weight)`` costs."""
    graph = nx.Graph()
    for u, v, w in pairs:
        graph.add_edge(u, v, weight=w, cost=edge_cost(w))
    return graph


# A chain of three genre cliques. Weights differ so the optimum is unique and
# the comparison against NetworkX cannot be decided by a tie-break.
CHAIN_PAIRS: list[tuple[str, str, float]] = [
    ("sf1", "sf2", 0.9),
    ("sf1", "sf3", 0.8),
    ("sf2", "sf3", 0.7),
    ("sf1", "gateway", 0.6),
    ("gateway", "zen", 0.5),
    ("zen", "phil1", 0.55),
    ("phil1", "phil2", 0.65),
    ("sf3", "lit1", 0.2),
    ("lit1", "lit2", 0.75),
    ("lit2", "phil2", 0.15),
]


@pytest.fixture
def chain():
    return make_projection(CHAIN_PAIRS)


@pytest.fixture
def planted_bridge_pairs() -> list[tuple[str, str, float]]:
    """Two cliques joined by exactly one node.

    Every route between the halves must pass through ``bridge``; a skeleton that
    misses it has not found the connecting structure at all.
    """
    left = ["sf1", "sf2", "sf3", "sf4"]
    right = ["lit1", "lit2", "lit3", "lit4"]
    pairs: list[tuple[str, str, float]] = []
    for group in (left, right):
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                pairs.append((a, b, 0.8))
    pairs.append(("sf1", "bridge", 0.5))
    pairs.append(("bridge", "lit1", 0.5))
    return pairs


class TestCostMatrix:
    def test_agrees_with_edge_cost_entry_by_entry(self, chain) -> None:
        """The vectorised transform must equal ``explain.edge_cost``.

        The matrix is built with a single NumPy ``log`` because a Python call per
        edge is seconds of work at 5M edges, so the two spellings are pinned
        together here the way ``fuse_in_sql`` is pinned to ``fuse_edges``.
        """
        costs = cost_matrix(chain).tocoo()
        adjacency = chain.adjacency.tocsr()
        for i, j, cost in zip(costs.row, costs.col, costs.data, strict=True):
            assert cost == pytest.approx(edge_cost(adjacency[int(i), int(j)]))

    def test_is_strictly_positive(self, chain) -> None:
        """Dijkstra rejects negative weights, and a zero is read as a non-edge."""
        assert (cost_matrix(chain).data > 0).all()

    def test_carries_no_infinities(self, chain) -> None:
        assert np.isfinite(cost_matrix(chain).data).all()


class TestMetricClosure:
    def test_leg_costs_match_networkx_shortest_path_length(self, chain) -> None:
        """The oracle for step 1-2 of KMB: distances over the *full* graph."""
        oracle = _nx_with_costs(CHAIN_PAIRS)
        result = connect_seeds(chain, ["sf2", "phil2"])
        assert len(result.skeletons) == 1
        for leg in result.skeletons[0].legs:
            expected = nx.shortest_path_length(
                oracle, leg.source, leg.target, weight="cost"
            )
            assert leg.cost == pytest.approx(expected)

    def test_leg_paths_are_real_shortest_paths(self, chain) -> None:
        oracle = _nx_with_costs(CHAIN_PAIRS)
        result = connect_seeds(chain, ["sf2", "phil2"])
        for leg in result.skeletons[0].legs:
            expected = nx.shortest_path(oracle, leg.source, leg.target, weight="cost")
            assert list(leg.path) == expected

    def test_strength_is_the_product_of_the_edge_weights(self, chain) -> None:
        """``exp(-cost)`` undoes the log, so a leg's strength is a probability."""
        result = connect_seeds(chain, ["sf2", "phil2"])
        weights = dict(((u, v), w) for u, v, w in CHAIN_PAIRS)
        for leg in result.skeletons[0].legs:
            product = 1.0
            for u, v in zip(leg.path, leg.path[1:], strict=False):
                product *= weights.get((u, v)) or weights[(v, u)]
            assert leg.strength == pytest.approx(product)


class TestSkeletonAgainstNetworkX:
    @pytest.mark.parametrize(
        "terminals",
        [
            ["sf2", "phil2"],
            ["sf2", "zen", "phil2"],
            ["sf1", "sf2", "lit2", "phil2"],
            ["sf3", "gateway", "phil1"],
        ],
    )
    def test_matches_the_networkx_steiner_tree(self, chain, terminals) -> None:
        """The oracle for the whole of KMB.

        NetworkX wants a graph object, which is why it cannot be the production
        path — building one over 392k nodes per request is not viable — but on a
        small graph it is exactly the right independent check.
        """
        oracle = nx.algorithms.approximation.steiner_tree(
            _nx_with_costs(CHAIN_PAIRS), terminals, weight="cost"
        )
        result = connect_seeds(chain, terminals)
        assert len(result.skeletons) == 1
        ours = {frozenset((u, v)) for u, v, _ in result.skeletons[0].edges}
        assert ours == {frozenset(edge) for edge in oracle.edges}

    def test_skeleton_is_a_tree(self, chain) -> None:
        result = connect_seeds(chain, ["sf2", "zen", "phil2"])
        skeleton = result.skeletons[0]
        assert len(skeleton.edges) == len(skeleton.nodes) - 1

    def test_edges_only_reference_skeleton_nodes(self, chain) -> None:
        skeleton = connect_seeds(chain, ["sf2", "lit2", "phil2"]).skeletons[0]
        nodes = set(skeleton.nodes)
        for src, dst, _ in skeleton.edges:
            assert src in nodes
            assert dst in nodes

    def test_no_non_terminal_leaves_survive(self, chain) -> None:
        """A dangling connector connects nothing; KMB's last step removes it."""
        skeleton = connect_seeds(chain, ["sf2", "phil2"]).skeletons[0]
        degree: dict[str, int] = dict.fromkeys(skeleton.nodes, 0)
        for src, dst, _ in skeleton.edges:
            degree[src] += 1
            degree[dst] += 1
        for node, count in degree.items():
            assert count > 1 or node in skeleton.terminals

    def test_terminals_and_connectors_partition_the_nodes(self, chain) -> None:
        skeleton = connect_seeds(chain, ["sf2", "phil2"]).skeletons[0]
        assert set(skeleton.terminals) | set(skeleton.connectors) == set(skeleton.nodes)
        assert not set(skeleton.terminals) & set(skeleton.connectors)

    def test_connectors_are_the_books_the_user_did_not_name(self, chain) -> None:
        skeleton = connect_seeds(chain, ["sf2", "phil2"]).skeletons[0]
        assert "gateway" in skeleton.connectors
        assert "zen" in skeleton.connectors


class TestPlantedBridge:
    def test_skeleton_contains_the_planted_bridge(self, planted_bridge_pairs) -> None:
        projection = make_projection(planted_bridge_pairs)
        result = connect_seeds(projection, ["sf2", "lit2"])
        assert "bridge" in result.skeletons[0].nodes
        assert "bridge" in result.skeletons[0].connectors

    def test_agrees_with_bridge_books(self, planted_bridge_pairs) -> None:
        """Two independent methods must identify the same joining book.

        ``bridge_books`` scores by the share of a node's edge weight that leaves
        its own community; the skeleton is found by shortest path over the whole
        graph. Nothing is shared between them but the graph, so their agreement is
        evidence rather than a tautology.
        """
        from bookmap.graph.centrality import bridge_books
        from bookmap.graph.communities import detect_communities

        projection = make_projection(planted_bridge_pairs)
        labels = detect_communities(projection)
        ranked = {work_id for work_id, _, _ in bridge_books(projection, labels, top_n=3)}

        skeleton = connect_seeds(projection, ["sf2", "lit2"]).skeletons[0]
        assert ranked & set(skeleton.nodes), (
            f"bridge_books said {ranked}, skeleton said {set(skeleton.nodes)}"
        )
        assert "bridge" in ranked
        assert "bridge" in skeleton.nodes


class TestDegenerateCases:
    """Every one of these occurs on the real graph."""

    def test_no_seeds_is_empty_not_a_crash(self, chain) -> None:
        result = connect_seeds(chain, [])
        assert isinstance(result, Connections)
        assert result.skeletons == ()

    def test_one_seed_has_nothing_to_connect(self, chain) -> None:
        result = connect_seeds(chain, ["sf1"])
        assert result.skeletons == ()
        assert result.terminals == ("sf1",)

    def test_repeated_seed_is_one_terminal(self, chain) -> None:
        result = connect_seeds(chain, ["sf1", "sf1"])
        assert result.terminals == ("sf1",)
        assert result.skeletons == ()

    def test_seed_absent_from_the_graph_is_reported_not_fatal(self, chain) -> None:
        result = connect_seeds(chain, ["sf2", "not-a-book", "phil2"])
        assert result.missing == ("not-a-book",)
        assert result.skeletons, "the reachable seeds should still be joined"

    def test_all_seeds_absent(self, chain) -> None:
        result = connect_seeds(chain, ["nope", "also-nope"])
        assert result.skeletons == ()
        assert set(result.missing) == {"nope", "also-nope"}

    def test_separate_components_get_one_skeleton_each(self) -> None:
        """The real graph is not connected, so this is the common case.

        Two joinable pairs in two components must both be answered, and neither
        may be silently folded into the other.
        """
        pairs = [
            ("a1", "a2", 0.5), ("a2", "a3", 0.5),
            ("b1", "b2", 0.5), ("b2", "b3", 0.5),
        ]
        projection = make_projection(pairs)
        result = connect_seeds(projection, ["a1", "a3", "b1", "b3"])
        assert len(result.skeletons) == 2
        grouped = {frozenset(s.terminals) for s in result.skeletons}
        assert grouped == {frozenset({"a1", "a3"}), frozenset({"b1", "b3"})}
        assert result.unjoined == ()

    def test_a_seed_alone_in_its_component_is_named(self) -> None:
        """Never a silent partial answer: say which seed could not be joined."""
        pairs = [("a1", "a2", 0.5), ("a2", "a3", 0.5), ("island", "islet", 0.5)]
        projection = make_projection(pairs)
        result = connect_seeds(projection, ["a1", "a3", "island"])
        assert len(result.skeletons) == 1
        assert set(result.skeletons[0].terminals) == {"a1", "a3"}
        assert result.unjoined == ("island",)

    def test_nothing_reachable_at_all(self) -> None:
        pairs = [("a1", "a2", 0.5), ("b1", "b2", 0.5)]
        projection = make_projection(pairs)
        result = connect_seeds(projection, ["a1", "b1"])
        assert result.skeletons == ()
        assert set(result.unjoined) == {"a1", "b1"}

    def test_isolated_node_in_the_graph_is_unjoined(self) -> None:
        """A resolved book with no fused edges cannot be routed to."""
        pairs = [("a1", "a2", 0.5)]
        projection = make_projection(pairs, nodes=["a1", "a2", "lonely"])
        result = connect_seeds(projection, ["a1", "lonely"])
        assert result.skeletons == ()
        assert "lonely" in result.unjoined


class TestHopCap:
    """A ten-hop chain through nine unknown books is not an insight."""

    @staticmethod
    def _long_chain(length: int) -> list[tuple[str, str, float]]:
        return [(f"n{i}", f"n{i + 1}", 0.5) for i in range(length)]

    def test_a_route_within_the_cap_is_returned(self) -> None:
        projection = make_projection(self._long_chain(3))
        result = connect_seeds(projection, ["n0", "n3"], max_hops=3)
        assert result.skeletons
        assert result.capped == ()

    def test_a_route_past_the_cap_is_refused_not_truncated(self) -> None:
        projection = make_projection(self._long_chain(8))
        result = connect_seeds(projection, ["n0", "n8"], max_hops=3)
        assert result.skeletons == (), "a truncated path is a lie about the route"
        assert [(c.source, c.target) for c in result.capped] == [("n0", "n8")]
        assert result.capped[0].hops == 8
        assert result.capped[0].limit == 3
        # Refused for length, not unreachable: these are different facts.
        assert set(result.unjoined) == {"n0", "n8"}

    def test_the_cap_is_reported_alongside_the_answer(self) -> None:
        result = connect_seeds(make_projection(self._long_chain(3)), ["n0", "n3"])
        assert result.max_hops == DEFAULT_MAX_HOPS

    def test_one_capped_pair_does_not_suppress_the_others(self) -> None:
        pairs = [*self._long_chain(9), ("n0", "near", 0.9)]
        projection = make_projection(pairs)
        result = connect_seeds(projection, ["n0", "near", "n9"], max_hops=2)
        assert len(result.skeletons) == 1
        assert set(result.skeletons[0].terminals) == {"n0", "near"}
        assert result.unjoined == ("n9",)
        assert result.capped


class TestWeightsAndCosts:
    def test_the_cheapest_route_is_the_strongest_chain_not_the_shortest(self) -> None:
        """``-log(weight)`` is chosen so summing costs multiplies probabilities.

        Two strong hops (0.9 * 0.9 = 0.81) must beat one weak direct link (0.2),
        which a plain hop count would get backwards.
        """
        pairs = [("a", "b", 0.2), ("a", "mid", 0.9), ("mid", "b", 0.9)]
        projection = make_projection(pairs)
        skeleton = connect_seeds(projection, ["a", "b"]).skeletons[0]
        assert "mid" in skeleton.nodes
        assert skeleton.legs[0].strength == pytest.approx(0.81)

    def test_a_direct_strong_link_needs_no_connector(self) -> None:
        pairs = [("a", "b", 0.9), ("a", "mid", 0.2), ("mid", "b", 0.2)]
        projection = make_projection(pairs)
        skeleton = connect_seeds(projection, ["a", "b"]).skeletons[0]
        assert skeleton.connectors == ()
        assert set(skeleton.nodes) == {"a", "b"}

    def test_total_cost_is_the_sum_of_the_skeleton_edges(self, chain) -> None:
        skeleton = connect_seeds(chain, ["sf2", "phil2"]).skeletons[0]
        expected = sum(edge_cost(weight) for _, _, weight in skeleton.edges)
        assert skeleton.cost == pytest.approx(expected)
        assert skeleton.strength == pytest.approx(math.exp(-expected))

    def test_zero_weight_entries_are_not_edges(self) -> None:
        """A stored zero is a non-edge, not a free ride between two books."""
        pairs = [("a", "b", 0.0), ("a", "mid", 0.5), ("mid", "b", 0.5)]
        projection = make_projection(pairs)
        skeleton = connect_seeds(projection, ["a", "b"]).skeletons[0]
        assert "mid" in skeleton.nodes


class TestDeterminism:
    def test_the_same_query_gives_the_same_skeleton(self, chain) -> None:
        first = connect_seeds(chain, ["sf2", "lit2", "phil2"])
        second = connect_seeds(chain, ["sf2", "lit2", "phil2"])
        assert first == second

    def test_seed_order_does_not_change_the_skeleton(self, chain) -> None:
        forward = connect_seeds(chain, ["sf2", "lit2", "phil2"]).skeletons[0]
        backward = connect_seeds(chain, ["phil2", "lit2", "sf2"]).skeletons[0]
        assert set(forward.nodes) == set(backward.nodes)
        assert {frozenset((u, v)) for u, v, _ in forward.edges} == {
            frozenset((u, v)) for u, v, _ in backward.edges
        }
