# CLAUDE.md

Orientation for working in this repo. Read `docs/todo.md` for open items and
`docs/plan.md` for why the design is shaped the way it is.

## What this is

`bookmap` turns Goodreads' "readers also enjoyed" lists and Amazon's "customers
also bought" data into a graph, then answers "given these books I like, what
else?" with network theory rather than a similarity lookup. Every recommendation
comes with a path back to a seed.

## Non-negotiables

- **uv only.** `uv run …`, `uv sync`. Never bare `pip` or `python`.
- **Test-driven, and the tests are the specification.** Write the failing test
  first. **Do not edit an existing test's assertions to make code pass** — if a
  test looks wrong, say so and explain why. Several real bugs in this project
  were found precisely because that rule was followed.
- Most components are checked against an **independent oracle**, not a snapshot
  of their own output: `networkx.pagerank` for PPR, a planted-partition graph for
  community detection, hand-computed arithmetic for fusion weights, real ISBN
  check digits for identity. Preserve that when adding tests.
- Comments explain **why**, not what.

```
uv run pytest                 # 531 tests, all green
uv run pytest -m slow         # scale tests (minutes; 20M-row synthetic graphs)
```

## Shape of the code

```
src/bookmap/
  models.py      frozen domain types; a node is a WORK, never an edition
  config.py      every tunable number lives here
  identity.py    ISBN/ASIN canonicalisation, fuzzy seed resolution
  store/db.py    DuckDB store — all SQL lives here
  store/projection.py   DuckDB -> Arrow -> scipy.sparse (never a Python row loop)
  sources/       one file per upstream dump; adapters know nothing about storage
  graph/         build (fusion), ppr, communities, centrality, layout, connect
  recommend.py   resolve seeds -> PPR -> hub damping -> filter -> MMR -> explain
  explain.py     shortest paths, cost = -log(weight)
  web/           FastAPI + a canvas renderer in static/
```

## Things that will bite you

- **Numbers in the Goodreads dump are strings, and often `""`.** Coerce
  defensively; never let a blank become `0`. `int_or_none` exists for this.
- **The dump separates `book_id` (edition) from `work_id` (work).** Nodes key on
  the work; each edition registers a `goodreads_id` alias so its edges land on
  the work node. Getting this wrong splits one book's signal across its printings
  — it did, and *Dune* appeared five times.
- **`similar_books` cites book_ids, including books outside the corpus.** Those
  edges are dropped by design. On a *slice* of the dump ~95% drop, which is
  correct, not a bug.
- **`ingest_sql` leaves `title_norm` NULL** (deriving `normalize_title` in SQL
  would drift from the Python definition). A backfill step fills it. Without it,
  bulk-ingested books are invisible to seed resolution.
- **Scale is the failure mode here, not logic.** Two shipped bugs were pipelines
  that worked perfectly on 177 books and did not terminate on 2.36M. If you touch
  ingest, fusion or resolution, add a `slow` test at realistic size.
- **MMR mixes two scales.** Candidate scores are normalised by their max before
  being traded against Jaccard redundancy; raw PPR mass is ~0.01 while Jaccard
  reaches 1.0, and without normalising, diversity swamps relevance entirely.
- **The web app opens the store read-only** so serving cannot collide with an
  ingest (DuckDB permits one writer). Keep it that way.
- **Colour in the map encodes role, not community** — only three hues clear the
  colour-vision floors for an all-pairs form. Community goes to position and
  labels. Run the palette validator before changing chart colours.

## Data

`data/demo/corpus.json` is **hand-authored** — real titles, designed edges. It is
labelled as such in the file and must never be presented as scraped Amazon or
Goodreads output. `data/dumps/` is gitignored; those files are multi-gigabyte and
research-use licensed.
