"""The minimal structure joining a set of seeds: a Steiner tree over the graph.

Seed *Dune*, *Emma* and *Zen and the Art of Motorcycle Maintenance* and the map
draws three separated lobes with nothing between them. That is not a layout
failure. The subgraph is assembled by **relevance** -- seeds, their top
recommendations, one hop of context -- and a book that links SF to Regency
romance is, by definition, not among the most similar books to either side. The
very nodes that would join the lobes are the ones selection excludes.

What is wanted is the *Steiner tree in graphs*: the minimum-weight subtree
spanning a set of terminals, free to route through intermediate nodes. Those
intermediates are the answer. It is NP-hard, so this is the standard
Kou-Markowsky-Berman 2-approximation:

1. Shortest paths from every terminal over the **full** graph, cost ``-log(w)``.
2. The *metric closure*: a complete graph over the terminals alone.
3. Minimum spanning tree of that closure -- with a handful of seeds, trivial.
4. Expand each closure edge back into the real path it stands for.
5. Union the paths, take an MST of the result, prune non-terminal leaves.

The expensive step is the first, and it is one C-level Dijkstra per terminal over
the whole graph. ``networkx.algorithms.approximation.steiner_tree`` is the right
*oracle* for this and the wrong implementation: it wants a graph object, and
building one over 392k nodes per request is not viable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree

from bookmap.explain import edge_cost

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.projection import GraphProjection

DEFAULT_MAX_HOPS = 8
"""Longest route reported as a connection between two seeds.

A ten-hop chain through nine books nobody has heard of is technically a path and
tells the reader nothing. Where the cap bites the pair is reported as *capped*
rather than truncated: a shortened path is a false claim about the route, while
"these two are further apart than N hops" is true and useful.

Raised from 6 after measuring the real graph: *Dune* and *Pride and Prejudice*
sit **7** hops apart, so a cap of 6 refused precisely the cross-genre query this
feature exists to answer. Eight admits those routes while still excluding the
ten-hop chains. The routes it admits are weak -- strength around 1e-4 -- but the
reported strength says so, which is better than declining to answer.
"""

_MIN_COST = 1e-12
"""Floor applied to every edge cost, for two mechanical reasons.

``scipy.sparse.csgraph`` reads a stored zero as a non-edge, so a weight of
exactly 1.0 -- cost zero -- would silently delete the strongest possible link;
and Dijkstra rejects negative weights, which a weight above 1.0 would produce.
Both are edge cases the fused weights make unlikely rather than impossible, and
the floor is far below any distance the result is compared at.
"""

_NO_PREDECESSOR = -9999
"""SciPy's sentinel in a predecessor matrix. Any negative value means the same."""

PRODUCT = "product"
"""Maximise the product of the edge weights: minimise the sum of ``-log(w)``.

The natural reading of "how probable is this chain of connections", and the
original objective. Its weakness, found on the real graph, is that it will trade
several strong links for one weak one — a single 0.107 edge costs 2.23 while three
0.25 hops cost 4.2 — so the cheapest route hangs off whichever tenuous edge
happens to exist.
"""

WIDEST = "widest"
"""Maximise the weakest edge on the route, then take the shortest such route.

Lexicographic, and it has to be. Pure maximin ignores length completely: on the
177-book demo corpus it routed *Dune* to *Emma* in **38 hops**, wandering the
spanning tree to avoid ever stepping down in weight. That answers "no tenuous
link" and abandons "how do these connect", which is trading one pathology for
another.

So this runs in two phases. The **maximum spanning tree** gives the optimal
bottleneck for each terminal pair exactly -- the minimax path between any two
nodes runs along it. The weakest of those bottlenecks then becomes an edge-weight
floor, and the ordinary product objective routes inside what survives. The result
carries no link weaker than strictly necessary, and among those is the strongest
chain.

Which makes this the same lever as ``min_edge_weight``, with the floor *derived
from the graph* rather than guessed: it is the highest floor that still joins
these particular seeds.
"""

OBJECTIVES = (PRODUCT, WIDEST)


@dataclass(frozen=True, slots=True)
class Leg:
    """One terminal-to-terminal segment of the route, with the books between."""

    source: str
    target: str
    path: tuple[str, ...]
    """Full node sequence, ``source`` first and ``target`` last."""

    cost: float
    """Sum of ``-log(weight)`` along the path."""

    strength: float
    """``exp(-cost)``: the product of the edge weights, so legs of different
    lengths are comparable as probabilities."""

    bottleneck: float = 0.0
    """The weakest single edge on the path.

    Reported for every objective because it is the number that exposes the
    weak-link artefact: a route can have a respectable product and still hang off
    one tenuous edge, and the product alone cannot tell you which happened. A
    2-hop leg at strength 0.047 reads as reasonable until you see that its
    bottleneck is 0.107.
    """

    @property
    def hops(self) -> int:
        return max(len(self.path) - 1, 0)

    @property
    def waypoints(self) -> tuple[str, ...]:
        """The books in the middle -- the ones the reader did not name."""
        return self.path[1:-1]


@dataclass(frozen=True, slots=True)
class Skeleton:
    """The connecting structure for one group of mutually reachable terminals."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, float], ...]
    """``(src, dst, weight)``, each undirected pair once, ``src < dst``."""

    terminals: tuple[str, ...]
    """Seeds this skeleton joins, in the order they were asked for."""

    connectors: tuple[str, ...]
    """Nodes the route passes *through* -- the books that do the joining."""

    legs: tuple[Leg, ...]
    cost: float

    @property
    def strength(self) -> float:
        return math.exp(-self.cost)


@dataclass(frozen=True, slots=True)
class CappedPair:
    """A pair whose cheapest route was refused for length. Reported, never hidden."""

    source: str
    target: str
    hops: int
    limit: int


@dataclass(frozen=True, slots=True)
class Connections:
    """The full answer, including everything that did *not* work.

    A partial answer presented as a whole one is the failure mode here: the real
    graph is disconnected, seeds go missing, and routes run long. Each of those
    gets its own field so a caller can say which happened.
    """

    skeletons: tuple[Skeleton, ...] = ()
    terminals: tuple[str, ...] = ()
    """Requested seeds that are present in the graph, deduplicated, in order."""

    missing: tuple[str, ...] = ()
    """Requested seeds with no node in the graph at all."""

    unjoined: tuple[str, ...] = ()
    """Present seeds that reached no other seed, by distance or by the hop cap."""

    capped: tuple[CappedPair, ...] = ()
    max_hops: int = DEFAULT_MAX_HOPS
    objective: str = PRODUCT
    """Which objective produced these routes. Reported so a caller comparing two
    runs cannot mix them up."""

    min_edge_weight: float = 0.0

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node on every skeleton, deduplicated, skeleton order preserved."""
        seen: dict[str, None] = {}
        for skeleton in self.skeletons:
            for node in skeleton.nodes:
                seen.setdefault(node, None)
        return tuple(seen)

    @property
    def edges(self) -> tuple[tuple[str, str, float], ...]:
        seen: dict[tuple[str, str], float] = {}
        for skeleton in self.skeletons:
            for src, dst, weight in skeleton.edges:
                seen[(src, dst)] = weight
        return tuple((src, dst, weight) for (src, dst), weight in seen.items())

    @property
    def connectors(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for skeleton in self.skeletons:
            for node in skeleton.connectors:
                seen.setdefault(node, None)
        return tuple(seen)

    @property
    def legs(self) -> tuple[Leg, ...]:
        """Every route across every group, for prose output and for tests.

        Flattened because a caller narrating "how do these connect" wants the
        routes, not the component structure they happen to fall into.
        """
        return tuple(leg for skeleton in self.skeletons for leg in skeleton.legs)


def cost_matrix(
    projection: GraphProjection, *, min_edge_weight: float = 0.0
) -> sp.csr_matrix:
    """The adjacency re-expressed as shortest-path costs, ``-log(weight)``.

    A vectorised restatement of :func:`bookmap.explain.edge_cost`, not a second
    definition of it: a Python call per stored entry is seconds of work at 5M
    edges, which is the whole budget for the query. The two are pinned to each
    other by test, the way ``fuse_in_sql`` is pinned to ``fuse_edges``.

    Non-positive entries are dropped rather than mapped to infinity. A stored
    zero is a non-edge (a projection assembled by summing duplicate edges can
    carry them), and an infinite entry in a ``csgraph`` is an impassable edge
    that still costs memory and comparisons on every relaxation.
    """
    adjacency = sp.csr_matrix(projection.adjacency).tocoo()
    # Dropped, not penalised. A floored edge must be genuinely impassable, or the
    # floor becomes a suggestion the router can overrule when nothing else works.
    keep = adjacency.data > max(0.0, min_edge_weight - _MIN_COST)
    costs = np.maximum(-np.log(adjacency.data[keep]), _MIN_COST)
    return sp.csr_matrix(
        (costs, (adjacency.row[keep], adjacency.col[keep])),
        shape=adjacency.shape,
    )


def _widest_costs(projection: GraphProjection, costs: sp.csr_matrix) -> sp.csr_matrix:
    """Restrict the graph to its maximum spanning tree.

    The minimax (widest) path between two nodes runs along the maximum spanning
    tree, so once the graph is restricted to that tree the unique remaining path
    *is* the widest path — no separate relaxation is needed and the rest of the
    pipeline is unchanged. On a disconnected graph this yields a forest, which is
    the right answer for a graph that genuinely has several components.

    SciPy only computes *minimum* spanning trees, so the weights are negated;
    ``costs`` is then reindexed onto the surviving edges, keeping ``-log(weight)``
    as the reported cost so that ``strength`` still means the same thing.
    """
    adjacency = sp.csr_matrix(projection.adjacency).tocoo()
    # Restricted to the edges the floor already admitted, so the two knobs
    # compose: a floored edge must not reappear via the tree.
    admitted = sp.csr_matrix(
        (np.ones(costs.nnz), costs.nonzero()), shape=costs.shape
    ).tocoo()
    live = {(int(i), int(j)) for i, j in zip(admitted.row, admitted.col, strict=True)}
    mask = np.fromiter(
        ((int(i), int(j)) in live for i, j in zip(adjacency.row, adjacency.col)),
        dtype=bool,
        count=adjacency.nnz,
    )
    negated = sp.csr_matrix(
        (-adjacency.data[mask], (adjacency.row[mask], adjacency.col[mask])),
        shape=adjacency.shape,
    )
    tree = minimum_spanning_tree(negated).tocoo()

    rows = np.concatenate([tree.row, tree.col])
    cols = np.concatenate([tree.col, tree.row])
    weights = np.concatenate([-tree.data, -tree.data])
    return sp.csr_matrix(
        (np.maximum(-np.log(weights), _MIN_COST), (rows, cols)), shape=adjacency.shape
    )


def _weakest_necessary(
    projection: GraphProjection, tree_costs: sp.csr_matrix, rows: np.ndarray
) -> float:
    """The lowest optimal-bottleneck across all terminal pairs, or 0.0 if none join.

    Phase one of :data:`WIDEST`. Each pair's tree path is bottleneck-optimal, so
    the weakest edge on it is the best that pair can do; the minimum over pairs is
    then the highest floor that still leaves every joinable pair joinable. Pairs in
    separate components contribute nothing -- they are unreachable at any floor.
    """
    adjacency = sp.csr_matrix(projection.adjacency)
    distances, predecessors = dijkstra(
        tree_costs, directed=False, indices=rows, return_predecessors=True
    )

    weakest = math.inf
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if not np.isfinite(distances[i][rows[j]]):
                continue
            walk = _walk(predecessors[i], int(rows[i]), int(rows[j]))
            if walk is None:
                continue
            weakest = min(
                weakest,
                min(
                    _weight(adjacency, u, v)
                    for u, v in zip(walk, walk[1:], strict=False)
                ),
            )
    return 0.0 if math.isinf(weakest) else weakest


def metric_closure(
    projection: GraphProjection,
    terminals: list[str] | tuple[str, ...],
) -> dict[tuple[str, str], float]:
    """Shortest-path cost between every ordered pair of terminals.

    Steps 1-2 of KMB, exposed on its own because the distances are worth reading
    without drawing anything: they say how far apart two genres are, in the same
    ``-log(weight)`` units the explanations use. Unreachable and unknown pairs
    are ``inf`` rather than absent -- a missing key makes "different components"
    indistinguishable from "never asked for", and callers compare rather than
    branch.
    """
    wanted: dict[str, int] = {}
    for work_id in terminals:
        row = projection.index.get(work_id)
        if row is not None:
            wanted.setdefault(work_id, row)

    closure = {
        (a, b): math.inf
        for a in dict.fromkeys(terminals)
        for b in dict.fromkeys(terminals)
        if a != b
    }
    if len(wanted) < 2:
        return closure

    rows = np.fromiter(wanted.values(), dtype=np.int64, count=len(wanted))
    distances = dijkstra(cost_matrix(projection), directed=False, indices=rows)
    for i, a in enumerate(wanted):
        for j, b in enumerate(wanted):
            if a != b:
                closure[(a, b)] = float(distances[i][rows[j]])
    return closure


def connect_seeds(
    projection: GraphProjection,
    terminals: list[str] | tuple[str, ...],
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    objective: str = PRODUCT,
    min_edge_weight: float = 0.0,
) -> Connections:
    """Find the minimal structure joining ``terminals``, via KMB.

    Returns one :class:`Skeleton` per group of mutually reachable terminals --
    the real graph is not connected, so several groups is the normal case, not an
    error -- plus the seeds that are missing, unjoinable, or too far apart.

    ``objective`` selects what "best route" means: :data:`PRODUCT` maximises the
    product of the edge weights, :data:`WIDEST` maximises the weakest edge on the
    route. ``min_edge_weight`` drops tenuous edges before routing and composes with
    either. Both exist because the product objective was measured on the real graph
    and found to route through single weak edges.
    """
    if objective not in OBJECTIVES:
        # Rejected rather than defaulted: a run reported under the wrong objective
        # is worse than one that failed, since the whole point is comparing them.
        raise ValueError(
            f"unknown objective {objective!r}; expected one of {OBJECTIVES}"
        )
    wanted: dict[str, int] = {}
    missing: list[str] = []
    for work_id in terminals:
        if work_id in wanted:
            continue  # the same book named twice is one terminal
        row = projection.index.get(work_id)
        if row is None:
            if work_id not in missing:
                missing.append(work_id)
        else:
            wanted[work_id] = row

    present = tuple(wanted)
    base = Connections(
        terminals=present,
        missing=tuple(missing),
        max_hops=max_hops,
        objective=objective,
        min_edge_weight=min_edge_weight,
    )
    if len(wanted) < 2:
        # Nothing to connect. Not an error: one seed is a legitimate query, it
        # simply has no route to anywhere else in it.
        return base
    if max_hops < 1:
        return Connections(**{**_fields(base), "unjoined": present})

    costs = cost_matrix(projection, min_edge_weight=min_edge_weight)
    floor = min_edge_weight
    if objective == WIDEST:
        rows = np.fromiter(wanted.values(), dtype=np.int64, count=len(wanted))
        floor = max(
            floor,
            _weakest_necessary(projection, _widest_costs(projection, costs), rows),
        )
        # Re-derive rather than mask the tree: the route must be free to leave the
        # spanning tree, which is what keeps it short once the floor is fixed.
        costs = cost_matrix(projection, min_edge_weight=floor)
    rows = np.fromiter(wanted.values(), dtype=np.int64, count=len(wanted))
    # One C-level pass per terminal over the whole graph. This is the only step
    # that touches all 392k nodes, and paths reconstruct from ``predecessors``
    # with no second traversal.
    distances, predecessors = dijkstra(
        costs, directed=False, indices=rows, return_predecessors=True
    )

    closure, paths, capped = _metric_closure(
        present, rows, distances, predecessors, max_hops=max_hops
    )

    n_groups, membership = connected_components(closure, directed=False)
    skeletons: list[Skeleton] = []
    unjoined: list[str] = []
    for group in range(n_groups):
        members = [i for i in range(len(present)) if membership[i] == group]
        if len(members) < 2:
            unjoined.append(present[members[0]])
            continue
        skeletons.append(
            _skeleton(projection, costs, present, members, closure, paths)
        )

    return Connections(
        skeletons=tuple(skeletons),
        terminals=present,
        missing=tuple(missing),
        # Request order, so the report reads in the order the user typed.
        unjoined=tuple(work_id for work_id in present if work_id in set(unjoined)),
        capped=tuple(capped),
        max_hops=max_hops,
        objective=objective,
        min_edge_weight=floor,
    )


def _fields(result: Connections) -> dict[str, object]:
    return {
        "skeletons": result.skeletons,
        "terminals": result.terminals,
        "missing": result.missing,
        "unjoined": result.unjoined,
        "capped": result.capped,
        "max_hops": result.max_hops,
        "objective": result.objective,
        "min_edge_weight": result.min_edge_weight,
    }


def _metric_closure(
    present: tuple[str, ...],
    rows: np.ndarray,
    distances: np.ndarray,
    predecessors: np.ndarray,
    *,
    max_hops: int,
) -> tuple[sp.csr_matrix, dict[tuple[int, int], list[int]], list[CappedPair]]:
    """Step 2: a complete graph over the terminals alone.

    Also returns the *realised* path behind each closure edge, since the
    predecessor matrix has to be walked anyway to count hops, and walking it
    twice for the same pair would be the only wasted work in the algorithm.
    """
    size = len(present)
    entries: list[tuple[int, int, float]] = []
    paths: dict[tuple[int, int], list[int]] = {}
    capped: list[CappedPair] = []

    for i in range(size):
        for j in range(i + 1, size):
            distance = float(distances[i][rows[j]])
            if not np.isfinite(distance):
                continue  # different components; reported by the caller
            walk = _walk(predecessors[i], int(rows[i]), int(rows[j]))
            if walk is None:
                continue
            hops = len(walk) - 1
            if hops > max_hops:
                # Refused, not shortened. The cheapest route within a hop budget
                # is *not* a prefix of the unbounded one, so a truncated path
                # would be a different -- and unverified -- claim.
                capped.append(CappedPair(present[i], present[j], hops, max_hops))
                continue
            entries.append((i, j, distance))
            paths[(i, j)] = walk

    if entries:
        i_index, j_index, values = (np.asarray(column) for column in zip(*entries, strict=True))
    else:
        i_index = j_index = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=float)
    closure = sp.csr_matrix(
        (
            np.concatenate([values, values]),
            (np.concatenate([i_index, j_index]), np.concatenate([j_index, i_index])),
        ),
        shape=(size, size),
    )
    return closure, paths, capped


def _walk(predecessors_row: np.ndarray, source: int, target: int) -> list[int] | None:
    """Rebuild ``source -> ... -> target`` from one row of the predecessor matrix."""
    path = [target]
    cursor = target
    # Bounded by the node count: a predecessor matrix is a tree, but a corrupt
    # one would otherwise spin forever.
    for _ in range(predecessors_row.shape[0]):
        if cursor == source:
            path.reverse()
            return path
        cursor = int(predecessors_row[cursor])
        if cursor < 0:
            return None
        path.append(cursor)
    return None


def _skeleton(
    projection: GraphProjection,
    costs: sp.csr_matrix,
    present: tuple[str, ...],
    members: list[int],
    closure: sp.csr_matrix,
    paths: dict[tuple[int, int], list[int]],
) -> Skeleton:
    """Steps 3-5 for one group: MST the closure, expand, union, MST, prune."""
    sub = closure[members, :][:, members]
    chosen = minimum_spanning_tree(sub).tocoo()

    work_ids = projection.work_ids
    adjacency = sp.csr_matrix(projection.adjacency)

    legs: list[Leg] = []
    union: dict[int, None] = {}
    for a, b in zip(chosen.row.tolist(), chosen.col.tolist(), strict=True):
        i, j = members[a], members[b]
        # ``paths`` is keyed on the lower index first; the route is undirected.
        walk = paths[(i, j)] if (i, j) in paths else list(reversed(paths[(j, i)]))
        for row in walk:
            union.setdefault(row, None)
        # The real fused weights, not the routed costs: under the widest objective
        # the router works over a spanning tree, but a leg's cost and strength must
        # describe the actual edges so the two objectives stay comparable.
        step_weights = [
            _weight(adjacency, u, v) for u, v in zip(walk, walk[1:], strict=False)
        ]
        cost = sum(edge_cost(weight) for weight in step_weights)
        legs.append(
            Leg(
                source=work_ids[walk[0]],
                target=work_ids[walk[-1]],
                path=tuple(work_ids[row] for row in walk),
                cost=cost,
                strength=math.exp(-cost),
                bottleneck=min(step_weights) if step_weights else 0.0,
            )
        )

    terminal_rows = {projection.index[present[i]] for i in members}
    tree = _prune(_union_mst(costs, list(union)), terminal_rows)

    on_tree = {row for edge in tree for row in edge}
    on_tree |= terminal_rows
    terminals = tuple(present[i] for i in sorted(members))
    connectors = tuple(
        work_ids[row] for row in sorted(on_tree - terminal_rows)
    )
    edges = tuple(
        sorted(
            (
                (work_ids[u], work_ids[v], _weight(adjacency, u, v))
                if work_ids[u] < work_ids[v]
                else (work_ids[v], work_ids[u], _weight(adjacency, u, v))
            )
            for u, v in tree
        )
    )
    return Skeleton(
        nodes=terminals + connectors,
        edges=edges,
        terminals=terminals,
        connectors=connectors,
        legs=tuple(sorted(legs, key=lambda leg: (leg.source, leg.target))),
        cost=sum(edge_cost(weight) for _, _, weight in edges),
    )


def _union_mst(costs: sp.csr_matrix, rows: list[int]) -> list[tuple[int, int]]:
    """Step 5's MST, over the union of the expanded paths.

    The union can contain cycles -- two legs may share a stretch of graph and
    rejoin -- and a Steiner *tree* must not, so the redundant edges go here.
    """
    if len(rows) < 2:
        return []
    index = np.asarray(rows, dtype=np.int64)
    induced = costs[index, :][:, index]
    tree = minimum_spanning_tree(induced).tocoo()
    return [
        (rows[int(a)], rows[int(b)])
        for a, b in zip(tree.row.tolist(), tree.col.tolist(), strict=True)
    ]


def _prune(edges: list[tuple[int, int]], terminals: set[int]) -> list[tuple[int, int]]:
    """Drop non-terminal leaves, repeatedly.

    A connector hanging off the end of the tree connects nothing -- it is an
    artefact of the union, not part of the answer -- and removing one can expose
    another, so this runs to a fixed point.
    """
    kept = list(edges)
    while True:
        degree: dict[int, int] = {}
        for u, v in kept:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        leaves = {
            node
            for node, count in degree.items()
            if count == 1 and node not in terminals
        }
        if not leaves:
            return kept
        kept = [(u, v) for u, v in kept if u not in leaves and v not in leaves]


def _weight(adjacency: sp.csr_matrix, u: int, v: int) -> float:
    """The real fused weight of one edge, for reporting rather than routing."""
    return float(adjacency[u, v])
