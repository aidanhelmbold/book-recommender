"""Explain a recommendation by the path that produced it.

"Because it is two hops from *Hyperion* and adjacent to *Blindsight*" is both
more convincing and more debuggable than a bare score. Explanations are also how
you notice the graph is wrong: a nonsense path means a bad edge.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np

from bookmap.models import Explanation

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.db import Store
    from bookmap.store.projection import GraphProjection

MAX_EXPANSIONS = 200_000
"""Ceiling on states the path search will pop before giving up.

Four hops out of a hub in the real graph can touch millions of nodes, and an
explanation is a nicety: degrading to fewer explanations is correct behaviour
where hanging the query is not.
"""


def edge_cost(weight: float) -> float:
    """Convert an edge weight to a shortest-path cost via ``-log(weight)``.

    Under this transform the lowest-cost path is the *highest-probability* chain
    of connections, since adding logs multiplies probabilities -- so a two-hop
    path through strong links can legitimately beat a weak direct link.
    """
    weight = float(weight)
    if weight <= 0.0:
        # An absent or zero-strength link is impassable, not an error: the cost
        # is genuinely infinite, and callers compare costs rather than branch.
        return math.inf
    return -math.log(weight)


def explain_paths(
    target: str,
    seeds: list[str],
    projection: GraphProjection,
    store: Store,
    *,
    max_paths: int = 3,
    max_hops: int = 4,
    max_expansions: int = MAX_EXPANSIONS,
) -> tuple[Explanation, ...]:
    """Find the strongest paths from distinct seeds to a recommendation.

    Returns at most ``max_paths`` explanations, each from a *different* seed
    (three paths from the same seed say little), strongest first. Paths longer
    than ``max_hops`` are omitted as unconvincing rather than padded.
    """
    if max_paths <= 0 or max_hops <= 0:
        return ()

    target_row = projection.index.get(target)
    if target_row is None:
        return ()

    # First spelling of a seed wins, and a seed that *is* the target is not an
    # explanation of itself.
    goals: dict[int, str] = {}
    for seed in seeds:
        row = projection.index.get(seed)
        if row is not None and row != target_row and row not in goals:
            goals[row] = seed
    if not goals:
        return ()

    paths = _cheapest_paths(
        target_row, set(goals), projection, max_hops=max_hops, max_expansions=max_expansions
    )
    if not paths:
        return ()

    titles = store.get_books(goals[row] for row in paths)
    work_ids = projection.work_ids

    explanations = []
    for row, (cost, rows) in paths.items():
        seed_id = goals[row]
        book = titles.get(seed_id)
        explanations.append(
            Explanation(
                seed_id=seed_id,
                seed_title=book.title if book is not None else seed_id,
                path=tuple(work_ids[step] for step in rows),
                # exp(-cost) undoes the log, so strength is the product of the
                # edge weights along the path -- a probability, comparable
                # between paths of different lengths.
                strength=math.exp(-cost),
            )
        )

    # seed_id tie-break so two equally strong paths always come back in the
    # same order; a reshuffling explanation reads as a changed recommendation.
    explanations.sort(key=lambda item: (-item.strength, item.seed_id))
    return tuple(explanations[:max_paths])


def _cheapest_paths(
    target_row: int,
    goals: set[int],
    projection: GraphProjection,
    *,
    max_hops: int,
    max_expansions: int,
) -> dict[int, tuple[float, list[int]]]:
    """Cheapest ``<= max_hops`` path from each goal row to ``target_row``.

    Searched outward from the target rather than inward from each seed: the
    graph is undirected, so one traversal serves every seed at once, and a
    recommendation typically has several seeds but only one target.

    Dijkstra over ``(row, hops)`` states rather than plain rows, because a hop
    limit makes the problem non-Markovian: the cheapest route to a node may use
    more hops than the budget left for the rest of the journey, so the shortest
    path within a hop bound is not a prefix of the unbounded one.
    """
    adjacency = projection.adjacency
    indptr, indices, data = adjacency.indptr, adjacency.indices, adjacency.data

    start = (target_row, 0)
    best: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    # (cost, hops, row): hops and row make the ordering total, so the traversal
    # is reproducible rather than dependent on heap tie-breaking.
    heap: list[tuple[float, int, int]] = [(0.0, 0, target_row)]

    found: dict[int, tuple[float, list[int]]] = {}
    expansions = 0

    while heap and len(found) < len(goals) and expansions < max_expansions:
        cost, hops, row = heapq.heappop(heap)
        state = (row, hops)
        if cost > best.get(state, math.inf):
            continue  # a cheaper route to this exact state was already settled
        expansions += 1

        if row in goals and row not in found:
            # Popped in cost order, so this is the cheapest path to this seed
            # within the hop budget. The parent chain runs seed -> ... -> target,
            # which is the direction an explanation is read in.
            walk = [row]
            cursor = state
            while cursor in parent:
                cursor = parent[cursor]
                walk.append(cursor[0])
            found[row] = (cost, walk)
            if len(found) == len(goals):
                break

        if hops >= max_hops:
            continue

        start_slice, end_slice = indptr[row], indptr[row + 1]
        for column, weight in zip(
            indices[start_slice:end_slice].tolist(),
            data[start_slice:end_slice].tolist(),
            strict=True,
        ):
            step = edge_cost(weight)
            if not np.isfinite(step):
                continue  # a zero-weight entry is a stored non-edge
            next_state = (column, hops + 1)
            next_cost = cost + step
            if next_cost < best.get(next_state, math.inf):
                best[next_state] = next_cost
                parent[next_state] = state
                heapq.heappush(heap, (next_cost, hops + 1, column))

    return found


def format_explanation(explanation: Explanation, titles: dict[str, str]) -> str:
    """Render an explanation as a human-readable line for CLI output."""
    seed_title = explanation.seed_title or titles.get(explanation.seed_id, explanation.seed_id)
    hops = explanation.hops
    if hops <= 0:
        return f"← {seed_title}"
    if hops == 1:
        # "1 hop" is a clumsy way to say "these two books are directly linked".
        return f"← directly linked to {seed_title}"

    # Naming the books walked through is what makes a path checkable by eye:
    # a stranger in the middle of the chain is how you spot a bad edge.
    waypoints = ", ".join(
        titles.get(work_id, work_id) for work_id in explanation.path[1:-1]
    )
    return f"← {hops} hops from {seed_title} via {waypoints}"
