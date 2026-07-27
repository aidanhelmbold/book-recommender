# Open items

Honest state of the project. Everything below is either unfinished, unverified, or
a known compromise — the working parts are described in the README.

Ordered by what would bite you soonest.

## Blocking nothing, but unverified

### Full-dump run has never completed end to end
The pipeline has been verified on the bundled 177-book corpus, on a 100k-record
slice of the real dump, and against 20M-row synthetic scale tests. It has **not**
been run to completion on the full 9.2 GB / ~2.36M-book file. Extrapolating from
the slice: ~11 minutes to ingest, ~17M raw edges, a 2–3 GB database.

Two things to watch on that first run, both flagged rather than solved:
- **Peak memory during `build`.** The graph projection is roughly 4–6 GB at full
  scale. If it is killed, raise `--min-weight` to prune weak edges *before*
  projection rather than fighting the allocation.
- **Leiden at ~2.4M nodes.** If community detection dominates the runtime, report
  it. Do not swap in a different algorithm to make it fast —
  `tests/test_communities.py` pins recovery of a planted partition, and that
  guarantee is the reason the clusters mean anything.

### Verification after the edition fix
Seeding `"Dune"` against a freshly ingested full dump should report **one** match
with no "also matched" list. If several still appear, work-level collapse is not
working on real data.

## Known gaps

### `--min-ratings` needs judging on real data
Implemented as `bookmap build --min-ratings N`: an edge survives only if **both**
endpoints clear the threshold, so filtering a book also removes the edges pointing
at it rather than leaving dangling refs.

A book whose `ratings_count` is **unknown** is kept at any threshold. The dump
reports the count as a string and frequently as `""`, which the adapters coerce to
NULL, and reading "we do not know" as "none" would delete a large slice of the real
corpus while looking like a working filter. `--min-ratings 0` is therefore a no-op,
not a filter on NULL.

Two things still open:
- **It does nothing on the demo corpus.** All 177 hand-authored books have no
  rating count, so every threshold is a no-op there. A test pins that, because the
  alternative — emptying the graph — is the silent failure to guard against.
- **No threshold has been judged on the real graph.** Try `--min-ratings 50` and
  `500` and compare `bookmap stats` node counts and recommendation quality. Use
  `--max-rank 10` alongside it if the graph is still too large; rank decay means
  positions 11–50 contribute little.

### Community labels do not handle non-English stopwords
The real dump is heavily multilingual, and slice labels included bare `de` and
`la`. TF-IDF should suppress those once communities are large and numerous enough
to contrast against each other, so this may resolve itself on the full graph —
judge from real full-run labels before adding a stopword list, since a hardcoded
English list would be the wrong fix for a multilingual corpus.

### Connection routes take one weak edge over a chain of strong ones
This is `docs/plan-connections.md` phase 4, now measurable. The plan predicted
routes would degenerate into **hub shortcuts** — a megaseller adjacent to
everything. On the demo corpus that is not what happens; something adjacent is:

```
Dune → Revelation Space → Atomic Habits   (2 hops, strength 0.0228)
Emma → The Duchess Deal  → Atomic Habits  (2 hops, strength 0.0163)
```

Both routes are arithmetically correct — `0.214 × 0.1069 = 0.0229`, and the edges
are real hand-authored corpus edges — and neither connector is a hub (degree 8 and
10, against Sapiens at 58). What is happening is that **one weak edge is cheaper
than several strong ones**: `-log(0.107)` is 2.23, while three 0.25-weight hops
cost 4.2. So the cheapest route hangs off whichever single tenuous link exists,
and *Revelation Space* is named not because it connects SF to self-help but
because it happens to carry the one edge that crosses.

The two-seed case reads much better (`Dune → The Dispossessed → Sapiens → Emma`,
where *The Dispossessed* is a planted bridge), so this is not uniformly wrong.

**Measured on the real 392k-node graph, and it survives.** The artefact is not a
small-corpus effect:

```
Dune → The Neutronium Alchemist (Night's Dawn, #2) → Foundation   (2 hops, 0.0473)
```

*Dune* and *Foundation* are two of the most-linked SF novels in the corpus and
there is no direct edge between them, so the route hangs off whichever single book
appears in both similar-books lists. Book two of a Peter F. Hamilton trilogy is
that book. It is a correct shortest path and a poor answer to "how do these
connect" — nobody reaches *Foundation* through the *Night's Dawn* sequence.

The three-way case shows the same shape more starkly: routing *Dune*, *Emma* and
*Zen and the Art of Motorcycle Maintenance* puts **The Book of Deeds of Arms and of
Chivalry** — an obscure medieval treatise — at the junction of both legs. The
literary stepping stones around it (*Bleak House*, *The Count of Monte Cristo*,
*Cat's Cradle*, *Cyrano de Bergerac*) read plausibly; the junction does not.

Note that **hub damping — the lever the plan proposed — is the wrong one here.**
This artefact is the opposite of a hub shortcut: the bad connectors are *low-degree*
books carrying one tenuous edge (*Revelation Space* has degree 8), so damping by
degree would push routes further toward them.

**Two levers are now implemented; which to adopt is still open.** Every route
reports its `weakest link`, and `--compare` runs all three on the same seeds:

- `--min-edge-weight 0.15` — refuse edges below a floor.
- `--objective widest` — maximise the weakest link, then take the shortest such
  route. Lexicographic by necessity: pure maximin ignores length and routed *Dune*
  to *Emma* in **38 hops** on the demo corpus. The maximum spanning tree gives each
  pair's best possible bottleneck and the weakest of those becomes the floor, which
  makes this the same lever with the threshold *derived* rather than guessed.

On the demo corpus both clearly improve the routes:

```
product          Dune → The Dispossessed → Sapiens → Emma
                 3 hops · strength 0.0012 · weakest link 0.063
widest           Dune → Do Androids Dream of Electric Sheep? → The Left Hand of
                 Darkness → Beloved → Persuasion → Emma
                 5 hops · strength 0.0007 · weakest link 0.189
```

The second is longer and lower-probability and is plainly the better answer, which
is the case against the product objective as a default.

**What is still needed:** the same comparison on the real 392k-node graph.
`bookmap route-report` gathers it in one command — a battery of probe seed sets
under all three objectives, with the resolved seed titles, the weakest link on each
route, and a per-objective timing, as markdown. Two things to watch there. First, whether `widest` derives a *usable* floor — on a
graph with 5.1M edges the weakest necessary link across a seed set may be so low
that the floor does nothing. Second, cost: `widest` adds a maximum spanning tree
over the whole graph plus a second Dijkstra pass, which is untimed at real scale
and may be too slow for a web request even if it is fine for the CLI. Neither
`--objective` nor `--min-edge-weight` changes any default yet, precisely because
that decision needs the real numbers.

Reported strength does at least make a weak route visibly weak (0.0473 against a
direct edge's 0.13–0.38), so nothing is hidden from the reader.

### A short title plus a truncated subtitle resolves to the wrong book
Typing `Atomic Habits` against the real graph resolves to **"Atomic: An I Bring the
Fire Short Story (A Loki Series)"**. The mechanism, and it is the same family as the
*Foundation* → back-pain-manual bug:

- `normalize_title` truncates at subtitle markers, so that title becomes `atomic`.
- `fuzz.WRatio("atomic habits", "atomic")` is **0.90**, well over the 0.75
  threshold, because WRatio matches partials so a query can find a long title.
- Nothing else scores higher, because *Atomic Habits* (2018) postdates the 2017
  UCSD dump and is genuinely **not in the corpus**.

So the honest answer was "no match" and the user got a confident wrong one. The
previous fix corrected which candidate wins a tie; this is a different defect — a
short truncated title matching a longer query at high confidence.

Candidate fix: penalise a candidate whose normalised title is a strict prefix of
the query but much shorter, or require the ratio to hold against the *untruncated*
title as well. Do not simply raise the threshold — 0.90 is high, and raising the
bar far enough to exclude this would start rejecting legitimate partial queries
like "harry potter". `tests/test_identity.py` has the ambiguous-title cases to
extend, and the failing example above belongs in them.

### The canvas renderer has no automated tests
`tests/test_web.py` covers every endpoint's contract and asserts that `map.js`
mentions the concepts it must handle, but nothing executes the renderer. The
drawing code — collision-avoided labels, diamond connectors, the emphasis pass for
the connections view — has only ever been verified by loading the page in
Chromium by hand and reading the screenshots in both themes.

That is how the label-overlap bug was found (`Emma` rendering as `Em` beneath
`The Duchess Deal`) and it would not have been caught any other way. `playwright`
is deliberately *not* a declared dependency, so those checks are manual and do not
run in the suite. `docs/plan-cicd.md` part 1 specs the `web` job that would fix
this; until it exists, changes to `map.js` need a browser and a pair of eyes.

### Amazon and Open Library paths are untested on real data
Both adapters are unit-tested against fixtures in the real formats, but no SNAP
`amazon-meta.txt`, Amazon Reviews 2023 file, or live Open Library call has ever
been run through them. The multi-source fusion weights (`config.SOURCE_ALPHA`)
have therefore never been exercised with more than one source present.

## Compromises worth knowing about

### Display title differs between the two ingest paths
When editions collapse onto one work, something has to choose the title. The bulk
SQL path picks the most-rated edition; the streaming path is last-write-wins. Both
are defensible and the two paths agree on the set of works and edges, which is the
invariant that matters — but they can disagree on the string shown.

### The map's third colour rarely appears
Nodes are coloured by role: seed / recommendation / context. With the default
result count, seed neighbours are usually already in the result set, so "context"
is often empty. The legend hides absent roles, so this is not misleading, but the
third slot earns its keep less often than expected.

### Diversity reranking is close to inert on the demo corpus
After the MMR scale fix, relevance dominates at the default `λ` = 0.7 and the
result set for SF seeds is pure SF — correct behaviour, but it means the demo data
no longer demonstrates the diversity pass end to end. The mechanism is unit-tested
where candidate scores are comparable.

### `resolve_refs` still exists for small sources
Bulk ingest now resolves endpoints inline, so resolution is only needed for the
streaming path, demo and `amazon-meta`. Resolving at ingest for those too would
let the function go away entirely; it is fast enough that this is an optimisation,
not a fix.

### `REF_ID_TYPE[DEMO]` names an alias nothing writes
`"demo_id"` has no corresponding `Book` field, so demo refs resolve via a
`books.work_id` fallback. It works and is tested, but the contract is odd. Either
have the demo ingest write its own aliases or drop the entry.

### Authors include audiobook narrators
The dump lists narrators among a book's authors — *Foundation* comes through as
"Isaac Asimov, Scott Brick". Harmless for display, but identity matching uses the
primary author, so ordering matters more than it looks.

## Housekeeping

- No pull request has been opened; all work is on
  `claude/book-recommendation-network-4dj2sq`.
- `uv run pytest -m slow` takes several minutes and builds 20M-row tables. It is
  excluded from the default run deliberately.
- No CI. A GitHub Actions workflow running `uv run pytest -m "not slow"` would be
  cheap and would have caught nothing so far, but keeps it that way.
