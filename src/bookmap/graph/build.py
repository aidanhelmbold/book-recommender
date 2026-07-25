"""Fuse typed, ranked, directed edges into one weighted undirected graph.

Two ideas do the work here.

**Position matters.** Being the first "customers also bought" result is a much
stronger statement than being the twentieth, so weight decays with list rank
instead of treating every listed neighbour alike.

**Sources disagree in quality.** Goodreads' similar-books list is a curated
recommender output; Amazon co-purchase is noisy (bundles, gifts, course
textbooks); shared subjects are merely topical. Each contributes at its own
trust level and the contributions add, so an edge attested by both Goodreads and
Amazon outranks one attested by either alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from math import log2

from bookmap.config import GraphConfig
from bookmap.models import Edge, EdgeKind, FusedEdge


def rank_weight(rank: int) -> float:
    """Positional decay: ``1 / log2(2 + rank)`` for 1-based ``rank``.

    Chosen over ``1/rank`` because it is gentler through the middle of a list:
    position 10 is worth roughly half of position 1 rather than a tenth, which
    matches how these lists actually degrade.
    """
    if rank < 1:
        raise ValueError(f"rank is 1-based, got {rank}")
    return 1.0 / log2(2 + rank)


def fuse_edges(
    edges: Iterable[Edge],
    config: GraphConfig | None = None,
) -> Iterator[FusedEdge]:
    """Collapse directed typed edges into undirected weighted ones.

    For each unordered pair::

        w(u,v)  = sum over kinds of  alpha[kind] * rank_weight(rank)
        weight  = max(w(u,v), w(v,u))
        dir_asym = 1 - min(w(u,v), w(v,u)) / max(w(u,v), w(v,u))

    ``max`` rather than a sum avoids double-counting a mutual pair as twice as
    strong as a one-way one, while ``dir_asym`` preserves the mutuality that
    would otherwise be discarded: 0.0 means both books list each other, 1.0
    means the link is one-way. Edges below ``config.min_weight`` and ranks
    beyond ``config.max_rank`` are dropped.

    Output is canonically ordered with ``src < dst`` so no pair appears twice.
    """
    config = config or GraphConfig()
    alpha = config.source_alpha

    # (src, dst) -> kind -> summed rank weight. Keeping the per-kind breakdown
    # instead of one running total means the summation order within a direction
    # depends only on which kinds are present, not on the order rows arrived in,
    # and it yields the ``kinds`` provenance tuple for free.
    directed: defaultdict[tuple[str, str], defaultdict[EdgeKind, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for edge in edges:
        if edge.rank > config.max_rank:
            continue
        if edge.kind not in alpha:
            continue
        directed[(edge.src, edge.dst)][edge.kind] += rank_weight(edge.rank)

    def direction_weight(kinds: dict[EdgeKind, float]) -> float:
        return sum(alpha[kind] * total for kind, total in sorted(kinds.items()))

    # Canonicalise to src < dst, then emit sorted so the output stream is
    # identical for any permutation of the input.
    pairs = {(src, dst) if src < dst else (dst, src) for src, dst in directed}

    for src, dst in sorted(pairs):
        forward = directed.get((src, dst), {})
        reverse = directed.get((dst, src), {})
        w_forward = direction_weight(forward)
        w_reverse = direction_weight(reverse)

        weight = max(w_forward, w_reverse)
        if weight <= 0.0 or weight < config.min_weight:
            continue

        yield FusedEdge(
            src=src,
            dst=dst,
            weight=weight,
            dir_asym=1.0 - min(w_forward, w_reverse) / weight,
            kinds=tuple(sorted(set(forward) | set(reverse))),
        )


def fuse_in_sql(store, config: GraphConfig | None = None) -> int:  # noqa: ANN001
    """Fuse directly inside DuckDB, bypassing Python entirely.

    Equivalent to :func:`fuse_edges` but expressed as a single aggregate query
    over ``edges_raw``; this is the path used for real dumps, where moving 20M
    edges through Python would dominate the build. Must agree with
    :func:`fuse_edges` to floating-point tolerance -- the tests hold both to the
    same hand-computed table.

    Returns the number of fused edges written.
    """
    config = config or GraphConfig()
    alpha = dict(config.source_alpha)
    conn = store.conn

    conn.execute("DELETE FROM edges_fused")
    if not alpha:
        return 0

    # A VALUES join rather than a CASE ladder: it supplies the per-kind trust
    # weight and drops kinds the config does not know about in one step.
    alpha_values = ", ".join(["(?, ?)"] * len(alpha))
    params: list[object] = []
    for kind, value in alpha.items():
        params.extend([str(kind), float(value)])
    params.extend([config.max_rank, config.min_weight])

    conn.execute(
        f"""
        INSERT INTO edges_fused (src, dst, weight, dir_asym, kinds)
        WITH directed AS (
            SELECT e.src AS src,
                   e.dst AS dst,
                   sum(a.alpha / log2(2 + e.rank)) AS w,
                   list(DISTINCT e.kind) AS kinds
            FROM edges_raw e
            JOIN (VALUES {alpha_values}) AS a(kind, alpha) ON e.kind = a.kind
            WHERE e.rank <= ?
            GROUP BY e.src, e.dst
        ),
        canonical AS (
            -- Fold each directed row onto its unordered pair, keeping the two
            -- directions in separate columns so max/min can be taken per pair.
            SELECT CASE WHEN src < dst THEN src ELSE dst END AS src,
                   CASE WHEN src < dst THEN dst ELSE src END AS dst,
                   CASE WHEN src < dst THEN w ELSE 0.0 END AS w_forward,
                   CASE WHEN src < dst THEN 0.0 ELSE w END AS w_reverse,
                   kinds
            FROM directed
        ),
        paired AS (
            SELECT src,
                   dst,
                   sum(w_forward) AS w_forward,
                   sum(w_reverse) AS w_reverse,
                   list_sort(list_distinct(flatten(list(kinds)))) AS kinds
            FROM canonical
            GROUP BY src, dst
        )
        SELECT src,
               dst,
               greatest(w_forward, w_reverse),
               1.0 - least(w_forward, w_reverse) / greatest(w_forward, w_reverse),
               kinds
        FROM paired
        WHERE greatest(w_forward, w_reverse) > 0.0
          AND greatest(w_forward, w_reverse) >= ?
        """,
        params,
    )
    row = conn.execute("SELECT count(*) FROM edges_fused").fetchone()
    return int(row[0]) if row else 0


def prune(
    edges: Iterable[FusedEdge],
    *,
    min_weight: float = 0.05,
    drop_leaves: bool = False,
) -> list[FusedEdge]:
    """Drop weak edges and optionally degree-1 nodes.

    Leaf dropping tightens the map but discards long-tail books, so it is off by
    default.
    """
    kept = [edge for edge in edges if edge.weight >= min_weight]
    if not drop_leaves:
        return kept

    degree: defaultdict[str, int] = defaultdict(int)
    for edge in kept:
        degree[edge.src] += 1
        degree[edge.dst] += 1

    # One pass, not a repeated peel to the 2-core: peeling would keep eating into
    # the long tail, which is exactly what this flag is meant to bound.
    return [edge for edge in kept if degree[edge.src] > 1 and degree[edge.dst] > 1]
