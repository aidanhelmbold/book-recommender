# Goal: get the real Goodreads graph ingested, built, and returning good recommendations

You are working in the `bookmap` repo on branch `claude/book-recommendation-network-4dj2sq`.
Python is managed by **uv** — always `uv run …`, never bare `pip` or `python`.

The pipeline is fully implemented and the test suite is green apart from the web
app. But **ingest of the real dump does not currently complete**: it stalls at
`resolving goodreads-ucsd edge endpoints…`. Phase 1 fixes that. Do not attempt
the full ingest before Phase 1 and Phase 2 pass.

## Inputs on this machine

```
data/dumps/goodreads_books.json          ~9.2 GB, newline-delimited JSON, NOT gzipped
data/dumps/goodreads_book_authors.json   ~106 MB, newline-delimited JSON
```

Both are read-only inputs. `data/dumps/` is gitignored — never commit them.

## Ground rules

- **Test-driven.** Every fix below lands its failing test *first*. Run the test,
  see it fail for the right reason, then implement until green.
- **Do not weaken a test to make it pass.** If a test looks wrong, say so and
  explain why rather than editing the assertion. The suite is the specification.
- Commit as you go on the existing branch. Do not push to the default branch and
  do not open a pull request unless asked.
- Check free disk before starting: the DuckDB file will likely reach several GB
  on top of the 9.2 GB input. Want ≥ 30 GB free and ideally ≥ 16 GB RAM.

```bash
uv sync
uv run pytest -q            # expect: green except ~26 in tests/test_web.py
df -h .
```

---

## Phase 1 — Fix the blockers

### 1.1 Write the scale test that should have caught this (do this first)

`tests/test_scale.py` guards PPR and the Arrow projection but **not** ingest or
ref resolution — the two steps that actually broke. That gap is why the whole
suite stayed green while the pipeline was unusable on real data.

Add to `tests/test_scale.py` (marked `slow`), before changing any source:

- Build a synthetic `edges_raw` of **20M+ rows** whose endpoints are namespaced
  refs (`goodreads_ucsd:<id>`), with a matching `books` table and `aliases`
  rows, roughly 10% of refs deliberately unresolvable.
- Assert `store.resolve_refs()` completes **within 120 s** and that the dropped
  count equals the unresolvable ones.
- Assert peak process RSS stays under a stated bound (e.g. 8 GB) — use
  `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`.

Run it. It should hang or blow the time limit. That is the bug reproduced.

```bash
uv run pytest tests/test_scale.py -m slow -q
```

### 1.2 Fix `resolve_refs`

`Store.resolve_refs` in `src/bookmap/store/db.py` (~line 282) is a **six-way
join over `edges_raw`** — order 40M rows for this dump (~2.36M books × ~20
`similar_books` each). Four of those joins key on *computed* expressions like
`substr(e.src, strpos(e.src, ':') + 1)`, so the indexes on `aliases` and `books`
cannot be used: DuckDB hash-probes multi-million-row tables with a freshly
derived key per row, per join, then materialises the entire resolved set into a
temp table. Instant on 177 books, unbounded on 40M.

Do not merely tune the query. Remove the need for it:

1. **Resolve at ingest, in the same pass.** `GoodreadsUCSDSource.ingest_sql`
   already reads the file with `read_json`. Have it join the unnested
   `similar_books` against its own staged book set inside that same query and
   write **already-canonical** `work_id` endpoints into `edges_raw`. Then
   `resolve_refs` has nothing to do for bulk sources and never sees 40M rows.
2. **Strip prefixes once.** Where resolution is still needed, project
   `src_raw`/`dst_raw` as real columns in one pass and join on those, so the keys
   are indexable — two joins, not six.
3. **Delete the `books` fallback from the hot path.** It exists only for the demo
   corpus and costs two of the six joins on every real row. Fix the underlying
   contract instead (1.3).
4. **Write back with one `CREATE TABLE … AS SELECT` + rename swap**, not a temp
   materialise followed by delete/insert.

Keep `resolve_refs` idempotent and keep its "drop unresolvable endpoints"
behaviour — `tests/test_store.py` and
`tests/test_sources_goodreads.py::TestIngestSQL` both pin it.

### 1.3 Fix the dead `demo_id` contract

`REF_ID_TYPE[SourceName.DEMO]` is `"demo_id"`, but nothing writes such an alias:
`Book` has no `demo_id` field and `DemoSource` has no `ingest_sql`. That is a
contract defect currently masked by the `books.work_id` fallback join you removed
in 1.2. Fix it properly — have the demo ingest write its own alias rows, or drop
the `DEMO` entry and make its ids explicitly canonical. Resolution should then be
uniform across sources with no special case.

`tests/test_integration.py` will catch a regression here immediately: if demo
edges stop resolving, the graph is 177 books and zero edges.

### 1.4 Key nodes by work, not edition

The dump carries **both** `book_id` and `work_id` (e.g. `5333265` → `5400751`).
The adapter currently mints `gr-<book_id>`, which is edition-level, so every
edition of *Dune* becomes a separate node splitting the similar-books signal the
recommender runs on — contrary to the design's "a node is a work, not an edition".

- Prefer the dump's `work_id`, falling back to `book_id` when it is absent/empty.
- `similar_books` cites **book_ids**, so each row must also register a
  `goodreads_id → work` alias, letting every edition's edges land on one node.
- Write the test first, with a fixture record carrying a distinct `work_id`.

Do this together with 1.2 — both change how `ingest_sql` writes ids.

### Phase 1 acceptance

```bash
uv run pytest -q                        # green except tests/test_web.py
uv run pytest tests/test_scale.py -m slow -q   # green, inside the time bound

rm -f /tmp/demo.duckdb
uv run bookmap ingest demo --db /tmp/demo.duckdb
uv run bookmap build --db /tmp/demo.duckdb
uv run bookmap recommend --seeds "Dune,Foundation,Hyperion" -n 10 --db /tmp/demo.duckdb
```

The demo recommendations must still be coherent science fiction with explanation
paths. Commit.

---

## Phase 2 — Dry run on a slice (do not skip)

Never debug a pipeline against a 9.2 GB file. Cut a slice, run the whole thing
end to end, measure, then extrapolate.

```bash
mkdir -p /tmp/slice
head -n 100000 data/dumps/goodreads_books.json > /tmp/slice/books.json
ls -la /tmp/slice/books.json      # expect roughly 400 MB

time uv run bookmap ingest goodreads-ucsd /tmp/slice/books.json \
    --authors data/dumps/goodreads_book_authors.json \
    --db /tmp/slice.duckdb
time uv run bookmap build --db /tmp/slice.duckdb
uv run bookmap stats --db /tmp/slice.duckdb
```

**Expect a low edge count relative to books, and say so rather than treating it
as a bug**: `similar_books` in a slice cites many books that are not in the
slice, and unresolvable endpoints are dropped by design.

Record wall-clock for ingest and for build. The full dump is ~24× this slice;
ingest should scale roughly linearly, `build` worse than linearly (community
detection and the projection both grow with edges). If the slice takes more than
about 5 minutes to ingest, stop and investigate before scaling up.

Then sanity-check that real titles resolve:

```bash
uv run bookmap recommend --seeds "Dune" -n 10 --db /tmp/slice.duckdb
```

If seeds do not resolve, check `title_norm` is populated — the bulk path fills it
via a backfill step, and books with `title_norm IS NULL` are invisible to the
seed resolver:

```bash
uv run python -c "
from bookmap.store.db import Store
with Store.open('/tmp/slice.duckdb', read_only=True) as s:
    print(s.counts())
    print('null title_norm:', s.conn.execute(
        'SELECT count(*) FROM books WHERE title_norm IS NULL').fetchone()[0])
"
```

## Phase 3 — Full ingest

```bash
rm -f bookmap.duckdb
time uv run bookmap ingest goodreads-ucsd data/dumps/goodreads_books.json \
    --authors data/dumps/goodreads_book_authors.json
```

While it runs, watch that RSS is stable rather than climbing without bound, and
that the DuckDB file is growing. Ingest is idempotent and records progress in
`ingest_state`, so an interrupted run can be repeated safely.

Checkpoint before building:

```bash
uv run python -c "
from bookmap.store.db import Store
with Store.open('bookmap.duckdb', read_only=True) as s: print(s.counts())
"
```

Expect books on the order of 2.3M (fewer if titleless records were skipped, and
fewer still than `book_id` count once editions collapse onto works) and
`edges_raw` in the tens of millions.

## Phase 4 — Build the graph

```bash
time uv run bookmap build --min-weight 0.05
```

This fuses edges in SQL, detects communities, labels them, and writes
`node_metrics`. Betweenness is off by default — leave it off at this scale.

Two things may need attention, and both are legitimate findings to report rather
than force through:

- **Memory during projection.** ~40M edges mirrored into COO then CSR is roughly
  4–6 GB. If it is killed, the honest fix is to raise `--min-weight` (pruning
  weak edges before projection) rather than to densify anything.
- **Leiden at ~2.4M nodes.** If community detection dominates the runtime, report
  the timing. Options are a coarser resolution, or pruning first — do not silently
  swap in a different algorithm, since `tests/test_communities.py` pins recovery
  of a planted partition.

## Phase 5 — Judge the output, not just the exit code

```bash
uv run bookmap stats
uv run bookmap recommend --seeds "Dune,Foundation,Hyperion" -n 20
uv run bookmap recommend --seeds "Pride and Prejudice,Emma" -n 20
uv run bookmap recommend --seeds "The Shining,Dracula" -n 20
uv run bookmap recommend --seeds "Atomic Habits,Deep Work" -n 20
```

For each, judge whether the results are actually plausible recommendations, and
report the real output. Specifically check:

- No seed appears in its own result list.
- Every recommendation carries an explanation path whose waypoints are **titles,
  not work ids** — a raw id means the titles map missed a node.
- SF seeds return SF, romance seeds return romance. A bestseller appearing under
  every unrelated seed set means hub damping is too weak; try `--beta 0.4`.
- Results are not all one author or one series. If they are, try
  `--exclude-same-author`.
- Multiple editions of the same book appearing separately means 1.4 did not work.

Then confirm determinism and the map export:

```bash
uv run bookmap recommend --seeds "Dune" -n 10 --json | shasum
uv run bookmap recommend --seeds "Dune" -n 10 --json | shasum   # identical
uv run bookmap map --seeds "Dune,Foundation" --out /tmp/map.html && open /tmp/map.html
```

## Failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| Stalls at `resolving … edge endpoints` | Phase 1.2 not applied | Stop; apply 1.2 |
| Seeds never resolve | `title_norm` NULL for bulk-ingested rows | Check the backfill ran; see Phase 2 query |
| Books ingest with no authors | `--authors` omitted | Re-run with the companion file |
| Graph has books but no edges | Refs all dropped in resolution | Check alias rows exist for `goodreads_id` |
| `build` killed by OOM | Projection memory | Raise `--min-weight`, rebuild |
| Recommendations are famous-but-unrelated | Hub damping too weak | Raise `--beta` |
| Same work appears as several nodes | Edition-level ids | Apply 1.4 |
| "no space left on device" | Disk | Delete stale `.duckdb`/`.npz`; deletes still succeed when writes fail |

## Report back

When done, summarise: the fixes made and their tests, slice vs full timings,
final `stats` output, and the actual recommendation lists with your honest
assessment of quality. If something is wrong and you could not fix it, say so
plainly and leave it failing rather than adjusting a test to hide it.
