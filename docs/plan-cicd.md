# Plan: CI and deployment

**Status:** proposed, not started. There is currently no `.github/` directory and
no automation of any kind.

## The constraint that shapes everything

**The data cannot go into CI, and it cannot go into a public deployment.** The
Goodreads and Amazon dumps are research-use licensed and multi-gigabyte. So:

- CI runs against the **bundled 177-book demo corpus** and synthetic graphs. That
  is enough — the demo corpus exercises the whole pipeline end to end, and the
  scale tests build their own data.
- Any publicly reachable instance can serve **only** the demo corpus. A public
  instance carrying the real graph would be redistributing a research dataset.
- A private instance carrying your real graph is fine, and is a different
  deployment topology with different rules. Both are covered below.

A second constraint follows from the architecture: **the database is an artifact
the code cannot produce for you.** A release ships code plus the demo corpus; the
real graph is built locally from dumps you obtained yourself. So "deploy" means
"ship the code and give it a way to attach a database", never "bake in a graph".

## Part 1 — CI

### Jobs

| Job | Trigger | Runtime | What it protects |
|---|---|---|---|
| **lint** | every push / PR | ~10s | `ruff check` + `ruff format --check` |
| **test** | every push / PR | ~40s | `uv run pytest -m "not slow"`, 414 tests |
| **pipeline** | every push / PR | ~30s | `ingest demo` → `build` → `recommend` → `bridges`, asserting real output |
| **web** | when `web/` or `recommend.py` changes | ~90s | Playwright against a real browser, both themes, console-error free |
| **scale** | nightly + manual | ~10min | `uv run pytest -m slow` |

Split this way because the first three must be fast enough that nobody learns to
skip them, while `scale` is minutes and builds 20M-row tables.

### Setup, the same on every job

```yaml
- uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
    cache-dependency-glob: uv.lock
- run: uv sync --frozen --all-groups
```

`--frozen` is the point: it fails if `uv.lock` is stale rather than silently
resolving something different from what anyone runs locally. `python-igraph` and
`leidenalg` build native extensions, so the uv cache matters — without it every
run pays for a compile.

### Version matrix

`requires-python = ">=3.11"`, so test **3.11, 3.12 and 3.13** on the `test` job
and pin the others to 3.12. The claim in `pyproject.toml` is currently untested;
either verify it or narrow it.

### Two things to fix first

**`ruff` is not a declared dependency.** It is used (`uv run ruff` resolves an
ambient 0.15.8) but appears nowhere in `dependency-groups`. CI would install a
different version from whatever a developer has and fail confusingly. Add it,
pinned, with an explicit `[tool.ruff]` config — the codebase already carries
`# noqa` codes implying rules nobody has written down.

**The scale job may not fit a runner.** `tests/test_scale.py` budgets
`RESOLVE_RSS_BUDGET_BYTES = 8 GB` and builds a 2M-book, 20M-edge table. A standard
GitHub-hosted runner gives 16 GB for public repositories but only ~7 GB for
private ones — under which the scale test fails on memory, not on a regression,
which is the worst kind of red build. Either confirm the runner size, add a
smaller CI-sized parameterisation, or run `scale` on a self-hosted runner. Decide
before wiring it, not after the first spurious failure.

### What CI would actually have caught here

Worth being concrete, because "add CI" is easy to cargo-cult:

- **Would have caught:** the CLI crash from touching a closed store — the
  `pipeline` job runs the real command and it failed outright. Also collection
  errors, import breakage, and lockfile drift.
- **Would NOT have caught:** the MMR scale bug (every test passed), the
  edition-splitting bug (no test existed), or the wrong-seed resolution (needed
  real data). Those were found by looking at real output.
- **Would have caught, but only if scheduled:** the `resolve_refs` non-termination
  — that is exactly what the `scale` job is for, and the reason it is worth the
  ten minutes nightly.

So CI is worth having and is not a substitute for running the thing on real data.
The `pipeline` job is the highest-value one, and it is the one most projects skip.

## Part 2 — Release

### PyPI, on tag

`uvx bookmap` and `uv tool install bookmap` are the natural distribution for a
local-first CLI. Use **trusted publishing** (OIDC) so no API token ever exists as
a secret.

```
on: push: tags: ['v*']
  → uv build
  → pypa/gh-action-pypi-publish   # id-token: write, no password
```

Gate the publish job on `lint`, `test` and `pipeline` passing. Version currently
lives literally in `pyproject.toml`; either adopt `hatch-vcs` so the tag *is* the
version, or add a CI check that the tag and the declared version agree — a
release whose metadata disagrees with its tag is a slow, confusing bug.

Ship the demo corpus inside the wheel (`data/demo/corpus.json`) so `bookmap ingest
demo` works from a bare install. It is ~200 KB and hand-authored by us, so there
is no licensing question. Verify it is actually included — a missing package-data
entry is the classic way this breaks, and only after publishing.

### Container image, on tag

For the VM route: build for `linux/amd64` and `linux/arm64`, push to GHCR.

**The image must not contain a graph.** It carries the code and the demo corpus;
a real database is mounted:

```
docker run -v $PWD/bookmap.duckdb:/data/bookmap.duckdb:ro \
           -p 8000:8000 ghcr.io/aidanhelmbold/bookmap web --db /data/bookmap.duckdb --host 0.0.0.0
```

Mounted read-only, which matches how the app already opens it.

## Part 3 — Deployment topologies

Three, with genuinely different rules.

**Local (today).** `uv run bookmap web`. No deployment needed. This is the primary
mode and should stay first-class.

**Private instance, real graph.** Your own VM or a Tailscale-reachable box. Build
the `.duckdb` where the dumps are, copy the ~2–3 GB artifact, run the container
against it. Blue/green is trivial and worth documenting: build the new database
beside the old, swap the symlink, restart — because DuckDB allows a single writer,
you cannot ingest into a live instance.

**Public demo, demo corpus only.** Fly.io or Render, tiny instance, built into the
image, clearly labelled as a hand-authored 177-book fixture and not real Goodreads
data. Deployed from `main` on green CI. This is the only public topology that is
licence-clean.

## Part 4 — What must be true before anything is public

Not a checklist for form's sake; these are real gaps in the current app.

- **`/api/subgraph` is an unauthenticated compute endpoint.** It runs ForceAtlas2
  at 500 iterations with O(n²) repulsion, per request, with `n` up to 400 chosen
  by the caller. That is trivially abusable. Before public exposure it needs a
  rate limit and a smaller hard cap, or a precomputed-layout path.
- **No CORS policy** is configured. Decide deliberately rather than inheriting a
  default.
- **No request logging or error reporting.** A public instance that fails silently
  cannot be debugged.
- **No auth on the private topology.** For a personal instance, put it behind
  Tailscale rather than inventing authentication.
- Static file serving already confines paths against traversal; keep the test that
  pins it.

None of these matter for local use, which is exactly why they are unaddressed —
worth stating so a future deployment does not assume otherwise.

## Part 5 — Phasing

1. **`lint` + `test` + `pipeline` on PR.** Add `ruff` to dev deps with a written
   config first. Highest value, one afternoon, no decisions required.
2. **`scale` nightly**, after resolving the runner-memory question.
3. **`web` job** with Playwright, when the map changes. Chromium is preinstalled in
   some environments but not on GitHub runners — budget for the install step.
4. **PyPI on tag**, with trusted publishing and the tag/version agreement check.
5. **Container image**, then the public demo instance if it is wanted.
6. The hardening in Part 4, gating anything internet-facing.

## Risks

- **Native builds make CI slower than it looks.** `python-igraph` and `leidenalg`
  compile if no wheel matches; cache the uv cache and pin the Python patch line.
- **The `web` job will be the flaky one.** Browser tests usually are. Keep its
  assertions structural — console errors, no horizontal overflow, an element
  exists — rather than pixel comparisons, and do not let it gate merges until it
  has proven stable.
- **A nightly `scale` job that nobody reads is theatre.** It needs to fail loudly
  into somewhere a human looks, or it is worse than not having it.
