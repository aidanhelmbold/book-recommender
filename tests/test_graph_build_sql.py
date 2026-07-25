"""``fuse_in_sql`` must agree with ``fuse_edges``, digit for digit.

Added because the two fusion paths had no test holding them together, and they
are trivially easy to drift apart: the Python path is what every test and the
demo ingest exercises, while the SQL path is what actually runs on a 20M-edge
dump. A divergence would silently reweight the entire production graph while the
whole suite stayed green.

Randomised rather than hand-tabulated on purpose -- the hand-computed table in
``test_graph_build.py`` already pins the arithmetic, so what is left to check is
*equivalence*, and equivalence is best attacked with awkward input: duplicate
kinds per pair, mutual and one-way pairs, ranks either side of ``max_rank``, and
weights either side of ``min_weight``.
"""

from __future__ import annotations

import random

import pytest

from bookmap.config import GraphConfig
from bookmap.graph.build import fuse_edges, fuse_in_sql
from bookmap.models import Edge, EdgeKind, SourceName

CONFIGS = [
    GraphConfig(),
    GraphConfig(max_rank=5, min_weight=0.0),
    GraphConfig(min_weight=0.3),
    GraphConfig(source_alpha={EdgeKind.GR_SIMILAR: 0.6}),
]


def _random_edges(n: int = 400, seed: int = 0) -> list[Edge]:
    rng = random.Random(seed)
    names = [f"w{i}" for i in range(30)]
    kinds = list(EdgeKind)
    sources = list(SourceName)
    seen: set[tuple[str, str, str, str]] = set()
    edges: list[Edge] = []
    for _ in range(n):
        src, dst = rng.sample(names, 2)
        kind = rng.choice(kinds)
        source = rng.choice(sources)
        # edges_raw is keyed on (src, dst, kind, source); duplicates could not
        # reach either fusion path, so generating them would test nothing.
        key = (src, dst, str(kind), str(source))
        if key in seen:
            continue
        seen.add(key)
        edges.append(Edge(src, dst, kind, rng.randint(1, 60), source))
    return edges


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"max_rank={c.max_rank}")
def test_sql_fusion_matches_python_fusion(store, config: GraphConfig) -> None:
    edges = _random_edges()
    store.insert_raw_edges(edges)

    written = fuse_in_sql(store, config)
    from_sql = {
        (edge.src, edge.dst): edge for edge in store.iter_fused_edges()
    }
    from_python = {(edge.src, edge.dst): edge for edge in fuse_edges(edges, config)}

    assert written == len(from_sql)
    assert set(from_sql) == set(from_python)
    for key, expected in from_python.items():
        got = from_sql[key]
        assert got.weight == pytest.approx(expected.weight, rel=1e-12)
        assert got.dir_asym == pytest.approx(expected.dir_asym, rel=1e-12, abs=1e-12)
        assert set(got.kinds) == set(expected.kinds)


def test_sql_fusion_replaces_rather_than_accumulates(store) -> None:
    """Rebuilding the graph must not double the weights of the previous build."""
    edges = _random_edges(n=50, seed=3)
    store.insert_raw_edges(edges)

    first = fuse_in_sql(store)
    before = {(e.src, e.dst): e.weight for e in store.iter_fused_edges()}
    second = fuse_in_sql(store)
    after = {(e.src, e.dst): e.weight for e in store.iter_fused_edges()}

    assert first == second
    assert before == after
