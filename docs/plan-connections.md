# Plan: show how unrelated genres actually connect

**Status:** proposed, not started.

## The question the map cannot currently answer

Seeding *Dune, Emma, Zen and the Art of Motorcycle Maintenance* draws three
separated lobes with nothing between them. The obvious question — *how do these
worlds join?* — is exactly the one the picture refuses to answer.

This is not a layout failure. **The subgraph is assembled by relevance, not by
connectivity:** seeds, their top recommendations, and one hop of context. A book
that links SF to Regency romance is, by definition, not among the most similar
books to either side — so the very nodes that would join the lobes are the ones
selection excludes. No force-directed tuning can draw an edge that was never
sent to the client.

## Why a plain minimum spanning tree is the wrong tool

Taking an MST *of the induced subgraph* is the natural reading of the idea, and it
does not work:

1. **It adds no nodes.** The connecting books still are not there. An MST over a
   disconnected node set yields a spanning *forest* — three separate trees, which
   is precisely the picture we already have, only sparser.
2. **It deletes real edges.** Within a genre these books form near-cliques, and
   that density is true and worth seeing. A tree would assert each cluster is a
   chain.
3. **It answers a different question.** MST minimises total edge weight across
   *everything*; what is wanted is the cheapest way to join a handful of specific
   terminals.

## What is actually wanted, and where the MST instinct is right

The question "what is the minimal structure connecting these particular nodes?"
is the **Steiner tree in graphs** problem — minimum-weight subtree spanning a set
of terminals, free to route through intermediate nodes. Those intermediates are
the answer: they are the books that join the genres.

It is NP-hard, but the standard 2-approximation (Kou–Markowsky–Berman) has an MST
at its heart, so the instinct lands in the right place:

1. Shortest paths from every seed over the **full** graph, cost `-log(weight)`.
2. Build the *metric closure*: a small complete graph over the seeds alone,
   edge weights = those shortest-path distances.
3. **Minimum spanning tree of that closure** — with 3–6 seeds this is trivial.
4. Expand each chosen closure edge back into the real path it stands for.
5. Union the paths, take an MST of the result, prune non-terminal leaves.

So: an MST, but over the *metric closure of the seeds*, not over the subgraph.
That distinction is the whole design.

The cost function already exists. `explain.edge_cost` is `-log(weight)`, chosen so
that summing costs multiplies probabilities — the cheapest path is the
highest-probability chain of connections, which is exactly the semantics wanted
for "how do these connect".

## Cost at real scale

Cheap, because the terminal count is tiny:

```python
from scipy.sparse.csgraph import dijkstra
distances, predecessors = dijkstra(
    cost_matrix, directed=False, indices=seed_rows, return_predecessors=True
)
```

One C-level Dijkstra per seed over 392k nodes / 5.1M edges — seconds, and paths
reconstruct from `predecessors` with no extra traversal. Steps 2–5 operate on a
graph with as many nodes as there are seeds.

`networkx.algorithms.approximation.steiner_tree` exists and is the right **test
oracle**, but it wants a NetworkX graph; building one over the full graph per
request is not viable, hence the SciPy path in production and NetworkX in the
tests.

## How it should appear on the map

**Overlay, not replacement.** Keep the relevance-driven neighbourhood and *add*
the skeleton nodes and edges. Two reasons: the recommendations are still the
primary output, and once real cross-lobe edges exist the force layout arranges
the clusters along the connecting spine by itself — the visual problem largely
solves itself once the data is honest.

- A view toggle: **Neighbourhood** (today) / **Connections** (skeleton emphasised,
  neighbourhood dimmed).
- Skeleton edges drawn heavier and at full opacity; everything else recedes.
- Connector nodes marked by **geometry, not colour** — the role palette is capped
  at three validated hues, and a fourth sits at ΔE 1.9 from blue under
  protanopia (see `docs/plan.md`). A ring is already taken by seeds and a double
  ring by bridges, so a distinct shape (diamond) is the remaining slot.
- A sentence naming the route in prose, because that is what people will quote:
  *"Dune → Gateway → Zen and the Art of Motorcycle Maintenance → Emma"*.

## It should agree with `bridges`, and that is testable

Bridge books are scored by cross-community edge weight; Steiner connectors are
found by shortest path. They are independent methods that should identify
overlapping books — and where they disagree, that is informative rather than
broken. On a graph with a **planted** bridge, both must find it. That gives a real
oracle rather than a snapshot:

| Component | Test |
|---|---|
| Metric closure | Distances match `nx.shortest_path_length` with `weight=cost` |
| Skeleton | Matches `nx.approximation.steiner_tree` on small graphs |
| Planted bridge | Two cliques joined by one node: the skeleton must contain it |
| Agreement | On the planted graph, `bridge_books` and the skeleton must overlap |
| Degenerate | One seed → empty skeleton; unreachable seeds → reported, not crashed |
| Scale | 392k-node graph, 5 seeds, within a stated time budget |

## Degenerate cases, all of which will occur

- **Seeds in different components.** The real graph is not connected. Return the
  skeleton for each reachable group and say plainly which seeds could not be
  joined — never a silent partial answer.
- **Very long routes.** A ten-hop chain through nine unknown books is not an
  insight. Cap the path length, and where the cap bites, say so rather than
  truncating invisibly.
- **A single seed.** No connections to show; fall back to the neighbourhood view.
- **Hub shortcuts.** The cheapest route between distant genres may run through a
  megaseller adjacent to everything, which is technically a path and tells you
  nothing. Worth testing whether applying the existing hub damping to path costs
  produces more meaningful routes — and worth *measuring* rather than assuming.

## Phasing

1. `graph/connect.py` — Dijkstra, metric closure, MST, path expansion. Pure graph
   code, tested against the NetworkX oracle. No UI.
2. `bookmap connect --seeds "A,B,C"` — print the route in prose. Cheap, and it
   makes the output judgeable before any pixels are involved.
3. `GET /api/connections` and the map overlay with the view toggle.
4. Evaluate the hub-shortcut question with real seeds and tune if warranted.

## Recommendation

Worth doing, and more valuable than it first appears: it turns the map from "here
are books near yours" into "here is the route between the things you like", which
is the thing a graph can say and a ranked list cannot. But do it as an **overlay
driven by a Steiner skeleton**, not as an MST of the current subgraph — the latter
would remove true structure and still leave the lobes apart.

The honest uncertainty is whether the routes read as *meaningful* on real data or
merely *short*. Phase 2 exists to answer that in prose, for the price of an
afternoon, before any rendering work is committed.
