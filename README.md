# bookmap

Give it a set of books you like. It gives you back books that suit the set as a
whole — with a graph you can look at, and a reason for every recommendation.

Under the hood it builds a network of book-to-book connections from Goodreads'
"readers also enjoyed" lists and Amazon's "customers also bought" co-purchase
data, then answers queries with network theory rather than a similarity lookup.

```
$ uv run bookmap recommend --seeds "Dune,Foundation,Hyperion" -n 5

  #   title                        author            score
  1   The Left Hand of Darkness    Ursula K. Le Guin  0.052
      ← 1 hop from Foundation, 2 from Dune
  2   Blindsight                   Peter Watts        0.048
      ← 2 hops from Hyperion, 2 from Dune
  ...
```

## Why a graph

The obvious approach — take each seed, look up its similar books, merge the
lists — answers the wrong question. It finds books similar to *Dune*, and books
similar to *Foundation*, but nothing that is characteristically similar to the
combination. A graph lets you ask the better question directly: start random
walks from all your seeds at once and see where they concentrate.

That is personalized PageRank (random walk with restart), and it naturally
rewards books reachable from *several* seeds over books tied strongly to just
one. Two further passes fix its known failure modes:

- **Hub damping.** Undamped, PPR recommends bestsellers to everyone — a book
  connected to 4,000 others accumulates walk probability regardless of
  relevance. Dividing by `degree^β` corrects for that.
- **Diversity reranking.** MMR over graph neighbourhoods stops the list
  collapsing into the single densest region your seeds touch.

Because the recommendation *is* a path through a graph, every result comes with
its justification: `← 2 hops from Hyperion, adjacent to Blindsight`. That is also
how you spot a bad edge.

## Data sources

There is no live API for this data any more. Goodreads retired its public API in
December 2020, and Amazon's Product Advertising API no longer exposes
similar-items. The available routes are published research dumps:

| Source | What it contributes | Notes |
|---|---|---|
| [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads) | ~2.36M books with a native `similar_books` field — the Goodreads recommender's own output | Primary edge source. Research use; cite the papers. |
| [SNAP `amazon-meta.txt`](https://snap.stanford.edu/data/amazon-meta.html) | Co-purchase `similar` ASINs | Books only; most of the file is DVDs and music. |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | `also_buy` / `also_view` | Kept as distinct edge kinds — viewing together is much weaker evidence. |
| [Open Library](https://openlibrary.org/developers/api) | Subjects, covers, years, canonical work ids | Public, keyless. Metadata enrichment, not behavioural edges. |

**Licensing.** The Goodreads and Amazon dumps are released for research use.
Check each dataset's terms before using them for anything else. `bookmap` ships
no scraped data and includes no scraper — the dumps are the supported path.

Both dumps are keyed differently (Goodreads ids versus ASINs), so nodes are
canonicalised to *works* rather than editions; without that the fused graph would
be two disconnected components describing the same books.

## Install

```
uv sync
```

## Quickstart, no download required

A small hand-authored demo corpus ships with the repo so the whole pipeline runs
immediately:

```
uv run bookmap ingest demo
uv run bookmap build
uv run bookmap recommend --seeds "Dune,Foundation,Hyperion" -n 20
uv run bookmap web          # interactive map at http://127.0.0.1:8000
```

The demo corpus is **hand-authored, not scraped** — it is labelled as such in
`data/demo/corpus.json`. It exists to make the code runnable and testable, not
to stand in for real data.

## The real graph

```
uv run bookmap ingest goodreads-ucsd ~/Downloads/goodreads_books.json.gz \
    --authors ~/Downloads/goodreads_book_authors.json.gz
uv run bookmap ingest amazon-meta ~/Downloads/amazon-meta.txt.gz
uv run bookmap ingest openlibrary          # optional metadata enrichment
uv run bookmap build --min-weight 0.05
uv run bookmap web
```

Ingest is resumable and idempotent — re-running a dump replaces rows rather than
accumulating them, which matters because edge weights are sums over the raw edge
table.

## How connections are weighted

List position carries real signal: being the first "customers also bought"
result says far more than being the twentieth. And sources differ in quality —
Goodreads' similar-books list is a curated recommender output, while Amazon
co-purchase is noisy with bundles, gifts and course textbooks.

```
rank_w   = 1 / log2(2 + rank)
w(u,v)   = Σ_kinds  α[kind] · rank_w(rank)     α: gr_similar     0.6
weight   = max(w(u,v), w(v,u))                    az_also_bought 0.3
dir_asym = 1 − min/max                            az_also_viewed 0.1
                                                  ol_subject     0.05
```

`dir_asym` preserves mutuality, which would otherwise be thrown away: 0.0 means
both books list each other, 1.0 means the link runs one way only.

Every knob lives in `src/bookmap/config.py`.

## Storage

One DuckDB file. The workload is bulk-load plus full-table analytical scan, which
is DuckDB's shape: `read_json` ingests the gzipped dump directly,
`UNNEST(similar_books) WITH ORDINALITY` recovers list position without a Python
loop, fusion is a single `GROUP BY`, and projection to SciPy hands Arrow buffers
straight to NumPy instead of pulling 20M rows through a cursor on every build.

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
| `bookmap stats` | Graph size, degree distribution, communities |

## Development

Built test-first: every component is pinned by tests written before its
implementation, and most are checked against an independent oracle rather than a
snapshot — `networkx.pagerank` for PPR, a planted-partition graph for community
detection, hand-computed arithmetic for fusion weights, real ISBN check digits
for identity resolution.

```
uv run pytest              # unit, property, integration
uv run pytest -m slow      # scale sanity (500k-node synthetic graphs)
```
