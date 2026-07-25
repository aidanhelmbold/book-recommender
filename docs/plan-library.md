# Plan: persist a personal library and recommend from it

**Status:** proposed, not started. Written to be executed the same way the rest of
this project was — test-first, oracles over snapshots, and scale tested before it
ships.

## Context

Today the seed set is whatever you type: `--seeds "Dune,Foundation,Hyperion"`.
That is fine for exploring and useless as a standing recommender. The goal is to
store your actual library — hundreds of books, with ratings and dates — and
recommend against it.

Two things make this more than "a longer `--seeds` list", and both need designing
rather than discovering:

**A library is not a seed set.** Uniform seeding over 500 books returns the
centroid of your entire reading history. The books adjacent to *many* of your
taste clusters at once are, almost by construction, the broadly popular ones — so
the naive version answers "what is famous near you?" rather than "what should I
read next?". Hub damping helps but does not fix the shape of the question.

**A library makes quality measurable for the first time.** Every recommendation
this project has produced so far has been judged by eye. With a library you can
hold out 10% of what you have already read and ask whether the recommender finds
it. That converts "these look plausible" into recall@k, and it is the single
biggest reason to build this feature. See [Evaluation](#evaluation).

## The library is user data, and must outlive the graph

The graph is derived: it gets rebuilt whenever fusion parameters change, and
`rm bookmap.duckdb` is a normal thing to do. A library is typed-in, imported,
irreplaceable data. **They must not share a file.**

Store the library in its own `library.duckdb` and `ATTACH` it to the graph
database at query time. That keeps the lifecycles separate, makes the library
trivially backup-able, and means a graph rebuild cannot destroy it.

```sql
-- library.duckdb
CREATE TABLE entries (
    work_id      VARCHAR,        -- resolved node; NULL when unmatched
    raw_title    VARCHAR NOT NULL,
    raw_author   VARCHAR,
    isbn13       VARCHAR,
    goodreads_id VARCHAR,        -- edition id from an export, if present
    shelf        VARCHAR NOT NULL,   -- read | reading | to-read | abandoned
    rating       INTEGER,        -- 1-5, NULL if unrated
    date_read    DATE,
    date_added   DATE,
    read_count   INTEGER,
    source       VARCHAR NOT NULL,   -- goodreads_csv | storygraph_csv | manual | …
    imported_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (source, raw_title, raw_author)
);

CREATE TABLE imports (
    source VARCHAR, artifact VARCHAR, rows_seen BIGINT, rows_matched BIGINT,
    imported_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (source, artifact)
);
```

`raw_title`/`raw_author` are kept verbatim alongside the resolved `work_id`
because **an unmatched book must still be stored**. A library will always contain
books absent from the dump; discarding them means a later dump refresh cannot pick
them up, and it hides the size of the gap from the user. Re-resolution becomes a
cheap pass over rows where `work_id IS NULL`.

Privacy: this is personal data. It stays local, is never transmitted, and
`library.duckdb` goes in `.gitignore`.

## Import

| Format | Notes |
|---|---|
| **Goodreads CSV export** | The priority. Columns include `Book Id`, `Title`, `Author`, `ISBN13`, `My Rating`, `Date Read`, `Exclusive Shelf`, `Read Count`. |
| StoryGraph CSV | Common Goodreads alternative; different column names, same shape. |
| Plain text | One title per line, optionally `Title — Author`. The fallback that always works. |
| Calibre / LibraryThing CSV | Nice-to-have, same code path as StoryGraph. |

**The Goodreads export is close to a perfect join, and this is the design's luckiest
break.** Its `Book Id` column is the *edition* `book_id` — exactly the key already
registered in the `aliases` table by the work-level ingest. So resolution is an
alias lookup, not fuzzy title matching, and it lands on the correct work node even
though the export names an edition. Fall back through ISBN13, then fuzzy
title+author via `identity.resolve_seed`, only when the id is missing.

Resolution must report its own accuracy: `matched 431 of 468 books (92%)`, with
`bookmap library unmatched` listing the rest. A silent 60% match rate would
produce quietly bad recommendations forever.

`Exclusive Shelf` carries real meaning and must not be flattened:

- `read` — the signal.
- `to-read` — **not** signal, but must still be excluded from output. Being told to
  read what you have already queued is a waste of a slot.
- `currently-reading` — exclude from output, weak signal.
- abandoned / did-not-finish — *negative* signal (see below).

## Weighting: not every book you read counts the same

```
w(book) = rating_weight × recency_weight

rating_weight:  5★ → 1.0   4★ → 0.6   3★ → 0.25   unrated → 0.4   1-2★ → 0 (see below)
recency_weight: max(0.3, exp(-age_years / 5))
```

Recency is floored rather than decaying to zero: a book you loved fifteen years ago
still says something true about you, it just says less than last month's. Both
curves belong in `config.py` as tunables, and both should be defensible from the
hold-out evaluation rather than chosen by taste — that is what the evaluation
harness is *for*.

These weights feed `personalized_pagerank(seed_weights=…)`, which already accepts
them. No change to the algorithm.

## Faceted recommendations — the actual new capability

Rather than one list from 500 seeds, cluster the library and recommend **per taste
cluster**:

1. Take the induced subgraph over the library's matched works.
2. Run community detection on *that* subgraph — not the global communities. The
   point is to find the structure of **your** reading, which may split "SF" into
   golden-age and new-weird in a way the global partition does not.
3. For each cluster with at least ~5 books, run a weighted PPR seeded on that
   cluster alone, then damp and rerank as now.
4. Present results grouped and labelled by the books that produced them.

```
Because you read Dune, Hyperion and Blindsight (12 more):
  1  The Quantum Thief — Hannu Rajaniemi
  …

Because you read Emma, Persuasion and The Grand Sophy (7 more):
  1  Bringing Down the Duke — Evie Dunmore
```

This solves the degeneracy problem directly instead of fighting it: each PPR run
has a tight, coherent seed set, which is the regime the algorithm is good at. It
is also far more useful output — "because you like X" is actionable in a way a
flat ranked list is not.

Keep a `--flat` mode for the whole-library view, and expect it to look duller.

## Negative signal

1–2★ ratings and abandoned books are information, and the tempting move —
subtracting a second PPR run — is also the easy way to produce nonsense, because
disliking one space opera does not mean disliking the genre.

Ship it in two stages:

- **Stage 1 (safe):** disliked and abandoned books are excluded from output and
  contribute zero weight. Nothing is subtracted.
- **Stage 2 (opt-in, `--avoid`):** `score = ppr_liked − γ · ppr_disliked` with γ
  small (~0.3). Gate this behind the hold-out evaluation: if it does not improve
  recall@k, do not ship it.

## Exclusions

Everything in the library is excluded from recommendations — read, reading and
to-read alike. This is the most obvious possible failure mode and deserves its own
test, because it is currently only enforced for the handful of explicit seeds.

Optional: `--include-owned` for people who want to be told what to re-read.

## Evaluation

The reason this feature is worth more than its convenience.

```
bookmap library evaluate [--holdout 0.1] [--k 20] [--seed 1917]
```

1. Split the `read` shelf into 90% seed / 10% hold-out, stratified by cluster so
   the split does not accidentally remove a whole taste.
2. Recommend from the seed portion.
3. Report **recall@k** (what fraction of held-out books appear in the top k),
   **MRR**, and a popularity baseline — recommend the highest-degree unread books
   and measure the same. *Beating the popularity baseline is the bar.* A
   recommender that cannot is an expensive way to say "read what's famous".

Then use it to tune, rather than guessing: `hub_beta`, `mmr_lambda`, the rating
and recency curves, and whether Stage 2 negative signal earns its place.

One caveat to state in the output: held-out books are books you *chose*, so this
measures agreement with your past choices, not discovery of things you would have
loved but never found. It is a real metric with a real blind spot.

## Surface

```
bookmap library import goodreads_library_export.csv
bookmap library import --format txt my-books.txt
bookmap library stats                     # counts by shelf, rating histogram, match rate
bookmap library unmatched                 # what failed to resolve, and why
bookmap library add "Piranesi" --rating 5 --shelf read
bookmap library remove "Piranesi"
bookmap library resolve                   # re-attempt matching after a dump refresh
bookmap library evaluate

bookmap recommend --from-library          # faceted by default
bookmap recommend --from-library --flat
bookmap recommend --from-library --shelf read --min-rating 4
```

Web: a library page (import, match rate, unmatched list), and a "from my library"
mode on the map where the user's own books are drawn as a distinct role — the
existing three-colour role encoding already has a slot for exactly this.

## Testing

Same discipline as the rest of the project; oracles where one exists.

| Component | Test approach |
|---|---|
| Goodreads CSV parser | A real export's header row, verbatim, including its quoting oddities and `=""1234""` id wrapping |
| Shelf semantics | Table-driven: each shelf value → expected weight and exclusion |
| Weighting | Hand-computed values, and monotonicity properties under `hypothesis` |
| Resolution ladder | A fixture exercising each rung: id hit, ISBN hit, fuzzy hit, no match |
| Exclusions | Property test: no library work_id ever appears in output, at any n |
| Faceting | A synthetic library planted in two known clusters must produce two facets |
| Evaluation harness | On a synthetic library where the answer is constructed, recall@k must be 1.0 |
| Scale | A 5,000-book library; import and faceted recommend within a stated time budget |

The exclusion property test matters most: it is cheap, and the bug it prevents is
the one a user notices in the first five seconds.

## Phasing

Each phase is independently useful and independently shippable.

1. **Store and import.** Schema, attach, Goodreads CSV, resolution ladder,
   `stats`/`unmatched`. No recommendation changes. Ends with a user able to see
   their library and its match rate.
2. **Flat library recommendations.** `--from-library` with weighting and full
   exclusion. Deliberately before faceting, so the degeneracy is observed rather
   than assumed.
3. **Evaluation harness.** Before tuning anything, so tuning has a target.
4. **Faceting.** The headline capability, tuned against phase 3.
5. **Web surface**, then optional negative signal if phase 3 says it helps.

## Risks

- **Match rate could be poor** for anyone whose library is not Goodreads-shaped.
  Mitigated by keeping raw titles and making `resolve` re-runnable — but a plain
  text list of translated titles will match badly and the tool must say so.
- **Faceting could over-split** a library into a dozen two-book clusters. Needs a
  minimum cluster size and a resolution parameter, tuned on real libraries.
- **The evaluation metric can be gamed** by recommending series continuations —
  held-out books are disproportionately next-in-series. Report recall with and
  without same-series matches, or the number flatters itself.
- **Cold start**: a library under ~20 matched books should fall back to flat
  recommendations and say why, rather than producing one facet per book.
