> **Status note.** This is the design plan the project was built from, kept as
> written so the reasoning behind each decision stays inspectable. Waves 0-3 and
> Wave 5 are complete; see `docs/todo.md` for what remains and for the places
> where the delivered system deliberately diverges from what is described here.
> The most significant divergence: the web map colours nodes by **role**, not by
> community, because only three hues clear the colour-vision separation floors
> for an all-pairs form and the graph has far more communities than that.

# Book Recommendation Network (`bookmap`)

## Context

The repo `aidanhelmbold/book-recommender` is empty. The goal: given a *set* of books as input, return books similar to that set as a whole — backed by a stored graph of book-to-book connections mined from Amazon "customers also bought" and Goodreads' similar-books data, explored through network theory.

Two hard facts shape the design:

1. **There is no live API for this data.** Goodreads retired its public API in Dec 2020; Amazon's PA-API 5.0 no longer exposes similar-items. The bulk-legal routes are the published research dumps: the **UCSD Book Graph** (Goodreads, ~2.36M books, carries a native `similar_books` field — literally the Goodreads recommender's own output) and **SNAP `amazon-meta.txt`** / **McAuley Amazon Reviews 2023** (`also_buy` / `also_view` co-purchase edges). Both are research-use licensed; the README will say so.
2. **This sandbox cannot reach them.** The org egress policy 403s every host except package registries — verified against `datarepo.eng.ucsd.edu`, `snap.stanford.edu`, `openlibrary.org`. Per the proxy README, policy denials are reported, not routed around.

So: build the entire system here and validate it end-to-end against a small **hand-authored demo corpus**, then the real graph loads with one `bookmap ingest` command on a machine with open egress. Only the input corpus is small — no algorithm is stubbed or faked. The demo fixture is written from my own knowledge of these books and will be labeled as such in-file, so it is never mistaken for scraped Amazon/Goodreads data.

Approved decisions: both dumps as edge sources + Open Library for metadata; hub-damped personalized PageRank with diversity reranking; local web app as the map UI; **uv** for packaging; **DuckDB** for storage.

### Why DuckDB over SQLite

This workload is bulk-load + analytical scan, which is exactly DuckDB's shape and exactly SQLite's weak spot:

- **Ingest becomes SQL.** `read_json('goodreads_books.json.gz', format='newline_delimited')` reads the gzipped dump natively, and `UNNEST(similar_books) WITH ORDINALITY` explodes the similar-books array *with its list position* in one statement. The Goodreads adapter collapses from a streaming Python loop to a single query; same for `amazon-meta`.
- **Fusion becomes SQL.** The rank-decay sum and `GREATEST` symmetrize are one aggregate with window functions, rather than a Python dict-merge over 20M edges.
- **Projection is near-free.** `.arrow()` → numpy → `scipy.sparse.csr_matrix` is zero-copy. Pulling 20M edges through a Python `sqlite3` cursor takes tens of seconds *on every build*; via Arrow it's sub-second.

The tradeoffs are real but don't bite here: DuckDB allows one writer (we ingest with a single process, and the web app opens the file `read_only=True`), and point lookups are slower than a B-tree — irrelevant at millisecond scale for typeahead. The store stays behind `store/db.py` so the backend remains swappable.

## Architecture

```
src/bookmap/
  models.py          Book, Edge, EdgeKind, SourceRef  (frozen dataclasses)
  config.py          source alphas, beta, lambda, PPR alpha — all tunable
  identity.py        canonical work IDs, ISBN10→13, ASIN↔ISBN, rapidfuzz title match
  store/
    schema.sql       books, edges, aliases, provenance, ingest_state
    db.py            DuckDB connection, DDL, upsert/query; resumable ingest
    projection.py    DuckDB → Arrow → scipy CSR + node index (cached .npz)
  sources/
    base.py          Source protocol: iter_books() / iter_edges()
    goodreads_ucsd.py  read_json on the .gz + UNNEST WITH ORDINALITY → ranked edges
    amazon_meta.py     SNAP amazon-meta.txt + McAuley also_buy/also_view
    openlibrary.py     async httpx: covers, subjects, canonical work IDs
    demo.py            bundled fixture loader
  graph/
    build.py         multi-source fusion, weighting, symmetrize, prune
    ppr.py           sparse power-iteration personalized PageRank
    communities.py   Louvain/Leiden + TF-IDF cluster labeling
    centrality.py    degree, betweenness, bridge-book detection
    layout.py        ForceAtlas2 on induced subgraphs
  recommend.py       seed resolve → PPR → hub damp → MMR → explain
  explain.py         seed→rec shortest paths (weight = −log w)
  web/app.py         FastAPI + static canvas frontend
data/demo/           curated fixture (~250 books, ~1500 edges)
tests/
```

**Stack:** Python 3.11 managed by **uv** (`pyproject.toml` + committed `uv.lock`, `uv sync`, `uv run bookmap …`; no bare `pip` anywhere, including in docs and CI). `duckdb`, `pyarrow`, `scipy`/`numpy` (sparse linear algebra), `networkx` (graph utilities + test oracle), `python-igraph` + `leidenalg` (community detection at 2M-node scale), `rapidfuzz`, `httpx`, `typer` + `rich`, `fastapi` + `uvicorn`, `pytest` + `hypothesis`.

### Storage: DuckDB, one file (`bookmap.duckdb`)

Node = **work**, not edition — collapsing editions is what makes ASIN and Goodreads-ID graphs joinable. `edges` is typed and carries provenance so any edge can be traced back to its source row, and `ingest_state` makes multi-GB ingests resumable. NetworkX is never the storage layer; it's a projection.

### Edge fusion (in SQL)

List position matters — the first "also bought" slot is a much stronger signal than the twentieth:

```
rank_w   = 1 / log2(2 + rank)             rank from UNNEST … WITH ORDINALITY
w(u,v)   = Σ_s α_s · rank_w_s(u,v)        α: gr_similar 0.6, az_also_bought 0.3,
                                             az_also_viewed 0.1, ol_subject 0.05
w_sym    = GREATEST(w(u,v), w(v,u))       original direction kept in `dir_asym`
```
One `GROUP BY` over the typed edge table, then prune below a weight floor and optionally drop degree-1 leaves.

### Recommendation

```
1. resolve seeds        title+author → work IDs (rapidfuzz; ambiguity surfaced, not guessed)
2. PPR                  p ← (1−α)·Mᵀp + α·s,  α=0.15, s uniform over seeds, L1 tol 1e-8
3. hub damping          score(v) = PPR_S(v) / deg(v)^β,  β=0.25
4. filter               drop seeds, optionally same-author / same-series
5. MMR rerank           argmax λ·score(v) − (1−λ)·max_{u∈sel} jaccard(N(v),N(u)),  λ=0.7
6. explain              2–3 shortest weighted paths back to distinct seeds
```

Step 3 is the quality lever: undamped PPR returns bestseller hubs for every seed set. Step 5 makes results span *all* clusters the seeds touch instead of only the densest one.

### Map (local web app)

`GET /` UI · `GET /api/search?q=` typeahead · `POST /api/recommend` · `POST /api/subgraph` · `GET /api/book/{id}/neighbors` (click-to-expand).

Layout is computed server-side per request on the **induced subgraph** (seeds + recs + 1-hop neighbors — hundreds of nodes, not millions); a full 2.36M-node layout is never attempted. Frontend is vanilla JS + canvas — no build step, no bundler. Communities colored, seeds ringed, recs sized by score, hover for title/author, click to expand. The app opens DuckDB `read_only=True` so serving can never collide with an ingest.

## Execution — test-driven

The tests are written **before** the implementation at every stage, and they are the executable form of the contracts above. This is what makes parallel sub-agents safe: each agent is handed a set of already-written failing tests that pin its module's behaviour, so "done" is defined before any agent starts guessing.

This graph code is unusually well suited to real TDD, because most of it has an independent oracle to assert against rather than a snapshot of its own output:

| Component | Test-first oracle |
|---|---|
| `graph/ppr.py` | `networkx.pagerank(..., personalization=…)` on small graphs — must agree to 1e-6 |
| `graph/communities.py` | **planted-partition** graph — must recover the known blocks (ARI > 0.9) |
| `graph/build.py` | rank-decay + fusion arithmetic worked out by hand in the test table |
| `identity.py` | real ISBN-10→13 check digits; known ASIN/ISBN pairs; ambiguous-title cases |
| `recommend.py` | hand-built graph where the correct hub-damped ranking is provable, not eyeballed |
| `graph/layout.py` | invariants, not coordinates — deterministic under fixed seed, finite, bounded, no overlap collapse |
| `sources/*` | few-line fixtures in genuine dump format, incl. a real `.gz`, with expected row counts |

### Wave 0 — me: scaffold + executable spec

`uv init`, `pyproject.toml`, then the shared contracts (`models.py`, `config.py`, `store/schema.sql`, `sources/base.py`) **and the full failing test suite** — fixtures, oracles, and one test module per component above, with implementations as bare stubs raising `NotImplementedError`.

Exit criterion: `uv run pytest` **collects cleanly and every test fails for the right reason** (unimplemented, not import errors or malformed fixtures). That red suite is the handoff artifact.

### Waves 1–3 — sub-agents, red → green

| Wave | Agent | Scope | Gate |
|---|---|---|---|
| 1 | A | `store/` (db, projection) + `identity.py` | its test modules green |
| 1 | B | `sources/` — four ingest adapters | its test modules green |
| 1 | C | `graph/` — ppr, communities, centrality, layout | its test modules green |
| 2 | D | `recommend.py`, `explain.py`, `cli.py` | + end-to-end CLI test green |
| 3 | E | `web/` — FastAPI + canvas frontend | + endpoint contract tests green |

Every agent's brief carries the same discipline: **run the failing tests first and confirm they fail, implement until green, do not edit a test to make it pass** — if a test looks wrong, report it to me instead of changing it. Any behaviour discovered mid-implementation that isn't yet pinned gets a *new* test written before the code that satisfies it.

Wave 1 runs in parallel (three agents, disjoint files, shared frozen contracts). Waves 2–3 are dependent. At each wave boundary I run the whole suite myself, review the diff, and integrate — a wave is not complete while any earlier wave's tests are red.

### Wave 5 — real-data remediation (current work)

Waves 0–2 are done and pushed; the web app is the only module still unimplemented. Running the real 9.2 GB UCSD dump surfaced three defects that the 177-book demo corpus could never expose. All three are mine, and all three are the same category of mistake: **a design validated only at toy scale.**

#### 5.1 `resolve_refs` does not terminate at real scale — blocking

`Store.resolve_refs` is a **six-way join over `edges_raw`** (order 40M rows for this dump: ~2.36M books × ~20 `similar_books` each). Four of those joins key on *computed* expressions — `substr(e.src, strpos(e.src, ':') + 1)` — so the indexes on `aliases` and `books` are unusable and DuckDB hash-probes multi-million-row tables with a freshly derived key per row, per join. The whole resolved set is then materialised into a temp table before anything is written back. At 177 books this is instantaneous; at 40M rows it is effectively unbounded, which is where the ingest is now stuck.

The fix is to stop treating resolution as a global post-pass over the edge table:

- **Resolve at ingest, in the same pass.** `GoodreadsUCSDSource.ingest_sql` already reads the file with `read_json`; it can join `similar_books` against its own staged book set inside that query and write **already-canonical** endpoints into `edges_raw`. `resolve_refs` then has nothing to do for the bulk sources, and never sees 40M rows.
- **Strip the prefix once, not four times.** Where resolution is still needed, derive `src_raw`/`dst_raw` in one projection and join on those plain columns, so the join keys are real columns that can be indexed — two joins, not six.
- **Delete the `books` fallback from the hot path.** It exists solely for the demo corpus and costs two of the six joins on every real row. Replace it by fixing the actual contract defect underneath it (5.2).
- Write back with a single `CREATE TABLE … AS SELECT` + swap rather than a temp-table materialise plus delete/insert.

#### 5.2 The `demo_id` contract is dead, and a join is papering over it

`REF_ID_TYPE[DEMO]` is `"demo_id"`, but nothing writes such an alias: `Book` has no `demo_id` field and `DemoSource` has no `ingest_sql`. That is a flaw in the Wave 0 contract, currently masked by the `books.work_id` fallback join that 5.1 needs to remove. Fix it at the source — have the demo ingest write its own alias rows (or drop the `DEMO` entry and let its ids be canonical explicitly) so resolution is uniform across sources and needs no special case.

#### 5.3 Nodes are keyed per edition, not per work

The real dump carries **both** `book_id` and `work_id` (e.g. `5333265` → `5400751`); the adapter currently mints `gr-<book_id>`, which is edition-level. So every edition of *Dune* is a separate node, splitting the similar-books signal that the recommender runs on — directly contrary to the stated design ("a node is a work, not an edition"). Prefer the dump's `work_id`, falling back to `book_id` when absent. Since `similar_books` cites **book_ids**, each ingested row must also register a `goodreads_id → work` alias so every edition's edges land on the one work node. Do this in the same change as 5.1, since both touch how `ingest_sql` writes ids.

#### 5.4 Close the scale-blindness in the suite itself

The real lesson is that `tests/test_scale.py` guarded PPR and projection but **not ingest or ref resolution** — the two steps that actually broke. Add scale coverage there, written first, asserting that resolving tens of millions of edges completes in bounded time and memory. A synthetic edge table is enough; the point is that this class of bug fails in CI rather than 40 minutes into a 9 GB run.

Same TDD discipline as every other wave: the failing test lands before the fix.

### Wave 4 — me: demo corpus, docs, and the tests that need the real system

The hand-authored demo corpus and README come last, because the integration and scale tests need working components to run against. Same rule: test before content.

## Verification

Layered, and each layer exists before the code it covers:

- **Unit** — the oracle table above, per module.
- **Integration** — `uv run bookmap ingest demo && uv run bookmap build && uv run bookmap recommend --seeds "Dune,Foundation,Hyperion" -n 20`: asserts SF/space-opera results, no seed echoed back, ≥3 communities represented, and every rec carries a non-empty explanation path that actually terminates at a seed.
- **Property-based** (`hypothesis`) where invariants beat examples: PPR mass always sums to 1 and stays non-negative; fusion weight is monotonically decreasing in rank; symmetrize is idempotent; ingest is idempotent — re-running it must not double edge weights (a genuine bug class here, worth pinning).
- **Web** — FastAPI `TestClient` over every endpoint, plus a malformed/unknown-seed path; manual load of the map to confirm rendering and click-expand.
- **Scale sanity** — synthetic 2M-node / 20M-edge graph confirming ingest, Arrow projection, and PPR stay in seconds with bounded memory. **Must also cover ref resolution and ingest**, the steps that actually broke on real data while the existing scale tests passed.

`uv run pytest` green plus the CLI walkthrough before anything is pushed to `claude/book-recommendation-network-4dj2sq`. Commits are structured so the failing tests land before or with their implementation, keeping the TDD history legible in the log.

Real-data ingest (your machine, open egress):
```
uv sync
uv run bookmap ingest goodreads-ucsd ~/Downloads/goodreads_books.json.gz
uv run bookmap ingest amazon-meta    ~/Downloads/amazon-meta.txt.gz
uv run bookmap build --min-weight 0.05
uv run bookmap web
```
