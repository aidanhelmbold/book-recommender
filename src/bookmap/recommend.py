"""The recommendation pipeline: a seed set in, a ranked explained list out.

    resolve seeds -> PPR over the whole set -> hub damping -> filter -> MMR -> explain

Each stage exists to fix a specific failure of the previous one. PPR alone
returns bestsellers; damping fixes that but leaves results clustered in one
genre; MMR spreads them across the regions the seeds actually touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bookmap.config import RecommendConfig
from bookmap.explain import explain_paths
from bookmap.graph.ppr import hub_damped_scores, personalized_pagerank
from bookmap.identity import normalize_author, resolve_seed
from bookmap.models import Recommendation, SeedMatch

if TYPE_CHECKING:  # pragma: no cover
    from bookmap.store.db import Store
    from bookmap.store.projection import GraphProjection

_SEARCH_LIMIT = 200
"""Rows the indexed title search offers the fuzzy matcher per seed query.

Seed resolution is fuzzy, but scoring every one of 2.4M titles per query is
seconds of work. ``title_norm`` is indexed, so a substring search narrows the
field first and the fuzzy pass only ranks what came back.
"""

_CANDIDATE_MULTIPLE = 50
_MIN_CANDIDATES = 500
"""How deep into the damped ranking MMR is allowed to look.

MMR is O(n * candidates), and the author filter needs a book row per candidate,
so neither can run over the whole graph. Reranking the top few hundred is the
standard retrieve-then-rerank split: diversity only ever promotes from within
the relevant set anyway.
"""


@dataclass(frozen=True, slots=True)
class RecommendResult:
    """Recommendations plus enough context to explain and draw them."""

    recommendations: tuple[Recommendation, ...]
    seeds: tuple[SeedMatch, ...]
    unresolved: tuple[str, ...] = ()
    """Queries that matched nothing -- surfaced, never silently dropped."""

    ambiguous: tuple[SeedMatch, ...] = ()
    """Queries with near-tied matches, for the caller to disambiguate."""


def resolve_seeds(
    queries: list[str],
    store: Store,
    *,
    threshold: float = 0.75,
) -> tuple[list[SeedMatch], list[str]]:
    """Resolve typed titles onto works. Returns ``(matches, unresolved)``."""
    matches: list[SeedMatch] = []
    unresolved: list[str] = []
    for query in queries:
        match = resolve_seed(query, _seed_candidates(query, store), threshold=threshold)
        if match.resolved:
            matches.append(match)
        else:
            # The query text, not the SeedMatch: the caller shows the user what
            # they typed back to them.
            unresolved.append(query)
    return matches, unresolved


def _seed_candidates(query: str, store: Store):  # noqa: ANN202 - iterable of candidate triples
    """Narrow the corpus to plausible matches for one seed query.

    Three tiers, cheapest first. The indexed substring search handles the normal
    case; splitting the query into tokens rescues a typo in *one* word of a
    multi-word title; the full scan is the last resort that guarantees a
    misspelling like "foundaton" still resolves. On a 2.4M-book corpus the first
    tier answers almost every real query.
    """
    found = store.search_titles(query, limit=_SEARCH_LIMIT)

    if not found:
        # Longest token first: it is the most selective, and the one least
        # likely to be the misspelled one.
        seen: set[str] = set()
        for token in sorted(query.split(), key=len, reverse=True):
            for book in store.search_titles(token, limit=_SEARCH_LIMIT):
                if book.work_id not in seen:
                    seen.add(book.work_id)
                    found.append(book)

    if found:
        return [(book.work_id, book.title, book.authors) for book in found]
    return store.title_candidates()


def mmr_rerank(
    candidate_rows: np.ndarray,
    scores: np.ndarray,
    projection: GraphProjection,
    *,
    n: int = 20,
    lambda_: float = 0.7,
) -> list[int]:
    """Greedy maximal-marginal-relevance selection over graph neighbourhoods.

    ``candidate_rows`` holds the projection rows eligible for selection.
    ``scores`` is a *full-graph* array aligned to ``projection.work_ids`` (the
    shape :func:`bookmap.graph.ppr.hub_damped_scores` returns), so no re-indexing
    is needed between stages. Returns the chosen rows in selection order.

    Selects the row maximising::

        lambda_ * score(v) - (1 - lambda_) * max_{u in selected} jaccard(N(v), N(u))

    Similarity is Jaccard overlap of adjacency, so two books with nearly the
    same neighbours are treated as redundant even if never directly linked --
    which is what stops a result set being five editions of one idea.
    """
    return [row for row, _ in _mmr_select(candidate_rows, scores, projection, n=n, lambda_=lambda_)]


def _mmr_select(
    candidate_rows: np.ndarray,
    scores: np.ndarray,
    projection: GraphProjection,
    *,
    n: int,
    lambda_: float,
) -> list[tuple[int, float]]:
    """MMR selection, returning ``(row, marginal value)`` in selection order.

    The value each row won its place with is kept because it is the only
    ranking number that is monotonically non-increasing down the list: a
    candidate's MMR value can only fall as more items are selected, so the
    greedy maxima come out in descending order. The raw damped score does not,
    since diversity deliberately promotes a lower-scoring row.
    """
    rows = [int(row) for row in candidate_rows]
    if not rows or n <= 0:
        return []

    scores = np.asarray(scores, dtype=float)

    # Rescale the candidate scores so the largest is 1.0, putting relevance on
    # the same [0, 1] footing as the Jaccard redundancy it is traded against.
    # Without this the two terms are incommensurable: damped PPR mass on a real
    # graph is ~0.01 while redundancy reaches 1.0, so the penalty outweighs
    # relevance roughly twentyfold and MMR stops ranking by relevance at all --
    # it selects whatever is most dissimilar, which is how a hard-SF seed set
    # ends up recommending philosophy. Dividing by the max rather than min-max
    # scaling keeps the *ratios* between candidates, so a runaway favourite
    # stays a favourite instead of being flattened toward its rivals.
    candidate_scores = scores[rows]
    largest = float(candidate_scores.max()) if candidate_scores.size else 0.0
    if largest > 0.0:
        scores = scores / largest

    adjacency = projection.adjacency
    indptr, indices, data = adjacency.indptr, adjacency.indices, adjacency.data

    def neighbourhood(row: int) -> frozenset[int]:
        start, end = indptr[row], indptr[row + 1]
        columns = indices[start:end]
        weights = data[start:end]
        # Explicitly stored zeros are not neighbours; see GraphProjection.degree.
        return frozenset(columns[weights != 0].tolist())

    neighbourhoods = {row: neighbourhood(row) for row in rows}

    remaining = set(rows)
    # Running max of jaccard(candidate, u) over everything selected so far, so
    # each step compares against one new selection rather than all of them.
    redundancy = dict.fromkeys(rows, 0.0)
    selected: list[tuple[int, float]] = []

    while remaining and len(selected) < min(n, len(rows)):
        # -row in the key breaks value ties toward the lower row index, which
        # fixes the output order for a given graph rather than leaving it to
        # set iteration.
        best = max(remaining, key=lambda row: (lambda_ * scores[row] - (1.0 - lambda_) * redundancy[row], -row))
        value = lambda_ * float(scores[best]) - (1.0 - lambda_) * redundancy[best]
        selected.append((best, value))
        remaining.discard(best)

        chosen = neighbourhoods[best]
        for row in remaining:
            overlap = _jaccard(neighbourhoods[row], chosen)
            if overlap > redundancy[row]:
                redundancy[row] = overlap

    return selected


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / (len(left) + len(right) - intersection)


def recommend(
    seed_queries: list[str],
    store: Store,
    projection: GraphProjection,
    *,
    n: int = 20,
    config: RecommendConfig | None = None,
) -> RecommendResult:
    """Run the full pipeline.

    Seeds themselves are always excluded from the output -- recommending a book
    the user already listed is the most obvious possible failure.
    """
    config = config or RecommendConfig()
    matches, unresolved = resolve_seeds(seed_queries, store)
    ambiguous = tuple(match for match in matches if match.ambiguous)

    seed_ids = [match.work_id for match in matches if match.work_id is not None]
    # A resolved seed with no fused edges is a real book with no graph around
    # it; it cannot start a walk, but it stays in ``seeds`` because the user did
    # name it successfully.
    seed_rows = sorted({projection.index[work_id] for work_id in seed_ids if work_id in projection.index})
    if not seed_rows:
        return RecommendResult((), tuple(matches), tuple(unresolved), ambiguous)

    ppr = personalized_pagerank(
        projection.adjacency,
        seed_rows,
        alpha=config.ppr_alpha,
        tol=config.ppr_tol,
        max_iter=config.ppr_max_iter,
    )
    degree = projection.degree()
    scores = hub_damped_scores(ppr, degree, beta=config.hub_beta)

    candidate_rows = _candidate_rows(scores, ppr, degree, seed_rows, n=n)
    if candidate_rows.size == 0:
        return RecommendResult((), tuple(matches), tuple(unresolved), ambiguous)

    work_ids = projection.work_ids
    books = store.get_books(work_ids[row] for row in candidate_rows.tolist())
    if config.exclude_same_author:
        # Filtered before MMR, not after: dropping the rest of a series only
        # counts as discovery if something else is promoted into its place.
        candidate_rows = _drop_shared_authors(candidate_rows, work_ids, books, seed_ids, store)
        if candidate_rows.size == 0:
            return RecommendResult((), tuple(matches), tuple(unresolved), ambiguous)

    chosen = _mmr_select(
        candidate_rows, scores, projection, n=n, lambda_=config.mmr_lambda
    )

    recommendations = []
    for row, value in chosen:
        work_id = work_ids[row]
        book = books.get(work_id)
        recommendations.append(
            Recommendation(
                work_id=work_id,
                title=book.title if book is not None else work_id,
                authors=book.authors if book is not None else (),
                score=value,
                # Both kept so a surprising ranking can be read off the result
                # rather than reproduced: raw_ppr / degree**beta is the score
                # MMR started from.
                raw_ppr=float(ppr[row]),
                degree=int(degree[row]),
                community=store.node_community(work_id),
                explanations=explain_paths(
                    work_id,
                    seed_ids,
                    projection,
                    store,
                    max_paths=config.max_explanations,
                ),
            )
        )

    return RecommendResult(tuple(recommendations), tuple(matches), tuple(unresolved), ambiguous)


def _candidate_rows(
    scores: np.ndarray,
    ppr: np.ndarray,
    degree: np.ndarray,
    seed_rows: list[int],
    *,
    n: int,
) -> np.ndarray:
    """The rows MMR is allowed to choose from, deepest-scoring first.

    Requires positive PPR mass *and* positive degree: a node the walk never
    reached is not a recommendation, and a node with no edges can be neither
    explained nor drawn.
    """
    eligible = (ppr > 0.0) & (degree > 0.0)
    eligible[seed_rows] = False
    rows = np.flatnonzero(eligible)
    if rows.size == 0:
        return rows

    depth = max(n * _CANDIDATE_MULTIPLE, _MIN_CANDIDATES)
    if rows.size > depth:
        # argpartition, not a full sort: at graph scale this is the difference
        # between O(n) and O(n log n) over millions of rows.
        keep = np.argpartition(-scores[rows], depth - 1)[:depth]
        rows = rows[keep]
    # Sorted by descending score with a row tie-break, so MMR sees a
    # deterministic candidate order.
    return rows[np.lexsort((rows, -scores[rows]))]


def _drop_shared_authors(
    candidate_rows: np.ndarray,
    work_ids: tuple[str, ...],
    books: dict,
    seed_ids: list[str],
    store: Store,
) -> np.ndarray:
    """Drop candidates sharing an author with any seed.

    Names are compared through :func:`normalize_author` so "Herbert, Frank" and
    "Frank Herbert" are one author -- the two spellings both occur across the
    dumps, and a filter that missed the inverted form would leak the series it
    was asked to suppress.
    """
    seed_books = store.get_books(seed_ids)
    excluded = {
        key
        for book in seed_books.values()
        for author in book.authors
        if (key := normalize_author(author))
    }
    if not excluded:
        return candidate_rows

    keep = []
    for row in candidate_rows.tolist():
        book = books.get(work_ids[row])
        authors = {key for author in (book.authors if book else ()) if (key := normalize_author(author))}
        if not authors & excluded:
            keep.append(row)
    return np.asarray(keep, dtype=candidate_rows.dtype)
