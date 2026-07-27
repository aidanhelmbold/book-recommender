# bookmap

Give it a set of books you like. It gives you back books that suit the set as a
whole — with a graph you can look at, and a reason for every recommendation.

Under the hood it builds a network of book-to-book connections from Goodreads'
"readers also enjoyed" lists and Amazon's "customers also bought" data, then
answers queries with network theory rather than a similarity lookup.

```
$ uv run bookmap recommend --seeds "Dune,Hyperion" -n 4

  1  I, Robot — Isaac Asimov  0.7000
      ← directly linked to Hyperion
      ← 2 hops from Dune via Foundation and Empire
  2  Dune Messiah — Frank Herbert  0.6307
  3  Foundation — Isaac Asimov  0.5600
  4  Do Androids Dream of Electric Sheep? — Philip K. Dick  0.5526
      ← 3 hops from Hyperion via Second Foundation, Ubik
```

There is also an interactive map: `uv run bookmap web`.

**Status:** complete and green — 531 tests, including scale tests over 20M-edge
synthetic graphs. Verified on the bundled corpus, on a 100k-record slice of the
real dump, and at synthetic scale. **A full 9.2 GB run has not yet completed end
to end.** See `docs/todo.md` for open items and known compromises, and
`docs/plan.md` for the design reasoning.

## Why a graph

The obvious approach — take each seed, look up its similar books, merge the lists
— answers the wrong question. It finds books similar to *Dune*, and books similar
to *Foundation*, but nothing characteristically similar to the combination. A
graph lets you ask the better question directly: start random walks from all your
seeds at once and see where they concentrate.

That is personalized PageRank, and it naturally rewards books reachable from
*several* seeds over books tied strongly to just one. Two further passes fix its
known failure modes:

- **Hub damping.** Undamped, PPR recommends bestsellers to everyone — a book
  connected to 4,000 others accumulates walk probability regardless of relevance.
  Dividing by `degree^β` corrects for it.
- **Diversity reranking.** MMR over graph neighbourhoods stops the list
  collapsing into the single densest region your seeds touch.

Because a recommendation *is* a path through a graph, every result carries its
justification — which is also how you spot a bad edge.

## Data sources

There is no live API for this data. Goodreads retired its public API in December
2020 and Amazon's Product Advertising API no longer exposes similar-items, so the
route is published research dumps.

| Source | Contributes | Notes |
|---|---|---|
| [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads) | ~2.36M books with a native `similar_books` field — the Goodreads recommender's own output | Primary edge source. Verified working. |
| [SNAP `amazon-meta.txt`](https://snap.stanford.edu/data/amazon-meta.html) | Co-purchase `similar` ASINs | Books only. Untested on real data. |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | `also_buy` / `also_view` | Distinct edge kinds — viewing together is far weaker evidence. Untested on real data. |
| [Open Library](https://openlibrary.org/developers/api) | Subjects, covers, years | Public, keyless. Enrichment, not behavioural edges. Untested against the live API. |

**Licensing.** The Goodreads and Amazon dumps are released for research use; check
each dataset's terms before other uses. `bookmap` ships no scraped data and
contains no scraper.

## Install

```bash
uv sync
```

## Quickstart, no download required

```bash
uv run bookmap ingest demo
uv run bookmap build
uv run bookmap recommend --seeds "Dune,Foundation,Hyperion" -n 20
uv run bookmap web          # http://127.0.0.1:8000
```

The demo corpus is **hand-authored, not scraped** — real titles, designed edges,
labelled as such in `data/demo/corpus.json`. It exists to make the code runnable
and testable, not to stand in for real data.

## The real graph

Full instructions, including a slice-first dry run, are in
[`docs/ingest-runbook.md`](docs/ingest-runbook.md). The short version:

```bash
mkdir -p /tmp/bookmap-spill

uv run bookmap ingest goodreads-ucsd data/dumps/goodreads_books.json.gz \
    --authors data/dumps/goodreads_book_authors.json.gz \
    --temp-dir /tmp/bookmap-spill --memory-limit 8GB --max-temp-size 20GB

uv run bookmap build --temp-dir /tmp/bookmap-spill --memory-limit 8GB --max-temp-size 20GB
uv run bookmap web
```

Expect roughly **11 minutes** to ingest and a **2–3 GB** database. Gzip the dumps
first — both readers and DuckDB handle `.gz`, and it saves about 7 GB of disk.

The spill controls matter. DuckDB's default temp directory sits beside the
database and its default cap is a fraction of the whole volume, so a query that
spills can fill the disk and take the machine with it; naming a directory and a
ceiling turns that into a query that fails quickly, saying which limit it hit.

Ingest is resumable and idempotent — re-running a dump replaces rows rather than
accumulating them, which matters because edge weights are sums over the raw table.

## How connections are weighted

List position carries real signal: being the first "customers also bought" result
says far more than being the twentieth. Sources also differ in quality — Goodreads'
similar-books list is a curated recommender output, while Amazon co-purchase is
noisy with bundles, gifts and course textbooks.

```
rank_w   = 1 / log2(2 + rank)
w(u,v)   = Σ_kinds  α[kind] · rank_w(rank)     α: gr_similar     0.6
weight   = max(w(u,v), w(v,u))                    az_also_bought 0.3
dir_asym = 1 − min/max                            az_also_viewed 0.1
                                                  ol_subject     0.05
```

`dir_asym` preserves mutuality that would otherwise be discarded: 0.0 means both
books list each other, 1.0 means the link runs one way only.

Every knob lives in `src/bookmap/config.py`.

## What sits *between* your books

Recommendation answers "what is near these". A graph can answer a question a
ranked list cannot: **how do these books connect to each other?**

```
$ uv run bookmap connect --seeds "Dune,Emma"
seed Dune — Frank Herbert
seed Emma — Jane Austen

Dune → The Dispossessed → Sapiens → Emma
  3 hops · strength 0.0012 · weakest link 0.063

connecting books
   1  Sapiens
   2  The Dispossessed
```

The books in the middle are the answer, and they are exactly the ones a similarity
search cannot return — a book joining SF to Regency romance is by definition not
among the most similar to either side. Finding them is the *Steiner tree in graphs*
problem: the minimum-weight subtree spanning a set of terminals, free to route
through intermediate nodes. It is NP-hard, so this uses the standard
Kou–Markowsky–Berman 2-approximation, with edge cost `-log(weight)` so that summing
costs multiplies probabilities and the cheapest route is the *strongest chain of
links* rather than merely the shortest one. Two strong hops beat one weak direct
edge, which a hop count would get backwards.

Because the real graph is disconnected, the result is one skeleton per group of
mutually reachable seeds, and seeds that could not be joined — or that turned out
further apart than the hop cap — are named rather than quietly omitted.

In the web app this is the **Connections** view: the skeleton is added to the map
(not substituted for it), its spine drawn heavy while the neighbourhood recedes,
and connecting books marked with a diamond. A diamond rather than a fourth colour
because the role palette is capped at three hues — only three clear the
colour-vision separation floors for a form where any two roles can end up adjacent
(see `docs/plan.md`), so identity that cannot rest on position rests on geometry.

### Choosing what "best route" means

Measured on the real graph, the default objective has a real weakness: maximising
the *product* of the edge weights will trade several strong links for one weak one,
because a single 0.107 edge costs less than three 0.25 hops. Routes then hang off
whichever tenuous link happens to exist — it put book two of a Peter F. Hamilton
trilogy between *Dune* and *Foundation*.

So every route reports its **weakest link** alongside its strength, and there are
two levers:

```
--min-edge-weight 0.15   refuse to route through edges weaker than this
--objective widest       maximise the weakest link, then take the shortest such route
```

`widest` is the same lever with the floor derived from the graph rather than
guessed: the maximum spanning tree gives each pair's best possible bottleneck, and
the weakest of those becomes the floor. It is lexicographic by necessity — pure
maximin ignores length and produced a 38-hop route on the demo corpus.

`--compare` runs all three on the same seeds so the difference is judgeable:

```
$ uv run bookmap connect --seeds "Dune,Emma" --compare
— product —
Dune → The Dispossessed → Sapiens → Emma
  3 hops · strength 0.0012 · weakest link 0.063

— widest —
Dune → Do Androids Dream of Electric Sheep? → The Left Hand of Darkness →
Beloved → Persuasion → Emma
  5 hops · strength 0.0007 · weakest link 0.189
```

The second is longer and lower-probability, and it is the better answer. Which one
is right in general is still open on real data; `docs/todo.md` carries the
measurements.

## Works, not editions

A node is a **work**. The dump carries both `book_id` (the edition) and `work_id`
(the work), and keying on the edition splits one book's evidence across every
printing of it — the real dump has five *Dune* editions, and each was collecting a
fifth of the signal. Nodes key on the work; each edition registers a
`goodreads_id` alias so its edges land on the one node.

## Storage

One DuckDB file. The workload is bulk-load plus full-table analytical scan, which
is DuckDB's shape: `read_json` ingests the gzipped dump directly,
`UNNEST(similar_books) WITH ORDINALITY` recovers list position without a Python
loop, fusion is a single `GROUP BY`, and projection to SciPy hands Arrow buffers
straight to NumPy — 2M edges in about a second.

Raw edges are kept immutable and separate from the fused graph, so fusion
parameters can be retuned and the graph rebuilt without re-ingesting anything.

## Commands

| Command | Purpose |
|---|---|
| `bookmap ingest <source> [path]` | Load a data source |
| `bookmap build` | Fuse edges, detect communities, compute metrics |
| `bookmap recommend --seeds "A,B,C"` | Ranked recommendations with explanations |
| `bookmap map --seeds "A,B" --out map.html` | Standalone HTML map |
| `bookmap web` | Interactive map server |
| `bookmap bridges` | Books linking two otherwise-separate reading communities |
| `bookmap connect --seeds "A,B,C"` | The route between books, and what joins them |
| `bookmap connect --compare` | The same route under each objective, side by side |
| `bookmap route-report` | A markdown report over a battery of seed sets |
| `bookmap stats` | Graph size, degree distribution, communities |

Write-side commands take `--temp-dir`, `--memory-limit` and `--max-temp-size`.
Read-side commands open the database read-only, so the web app is safe to leave
running during a re-ingest.

## Development

Built test-first, and most components are checked against an **independent
oracle** rather than a snapshot of their own output: `networkx.pagerank` for PPR,
a planted-partition graph for community detection, hand-computed arithmetic for
fusion weights, real ISBN check digits for identity resolution.

```bash
uv run pytest              # 531 tests
uv run pytest -m slow      # scale tests: 20M-row resolution, 500k-node PPR
```

See `CLAUDE.md` for conventions and the traps this dump sets.
