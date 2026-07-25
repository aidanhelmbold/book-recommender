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

### `--min-ratings` is not implemented
Intended as a way to drop the long tail — obscure books have thin, unreliable
`similar_books` edges, so filtering them yields a smaller graph and arguably
better recommendations. The agent implementing it was cut off before starting.

Workaround: `--max-rank 10` cuts edges roughly threefold, and rank decay means
positions 11–50 contribute little anyway. Use the same value for `ingest` **and**
`build`; a mismatch silently changes the graph.

When implementing: `ratings_count` arrives as a *string* and is frequently `""`.
An absent count must not be silently treated as 0 and excluded. Filtering must
also drop edges pointing at excluded books rather than leave dangling refs.

### Community labels do not handle non-English stopwords
The real dump is heavily multilingual, and slice labels included bare `de` and
`la`. TF-IDF should suppress those once communities are large and numerous enough
to contrast against each other, so this may resolve itself on the full graph —
judge from real full-run labels before adding a stopword list, since a hardcoded
English list would be the wrong fix for a multilingual corpus.

### Betweenness and bridge books are computed but never surfaced
`graph/centrality.py` implements `approximate_betweenness` and `bridge_books`, and
they are tested, but nothing in the CLI or the web app shows them. Bridge books —
titles linking two otherwise-separate reading communities — are arguably the most
interesting recommendation available and are currently invisible. `build
--betweenness` computes the metric; nothing reads it.

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
