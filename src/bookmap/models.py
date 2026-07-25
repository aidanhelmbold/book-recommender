"""Core domain types.

These are the contracts every other module binds against. A node is a *work*,
not an edition: collapsing editions is what makes the Amazon (ASIN-keyed) and
Goodreads (book-id-keyed) graphs joinable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EdgeKind(StrEnum):
    """Where a connection came from, and what it means.

    The kind determines the fusion weight (see ``config.SOURCE_ALPHA``), so the
    distinction between "bought together" and "merely viewed together" is
    preserved rather than flattened at ingest time.
    """

    GR_SIMILAR = "gr_similar"
    """Goodreads' own "readers also enjoyed" list."""

    AZ_ALSO_BOUGHT = "az_also_bought"
    """Amazon "customers who bought this also bought"."""

    AZ_ALSO_VIEWED = "az_also_viewed"
    """Amazon "customers who viewed this also viewed" — weaker signal."""

    OL_SUBJECT = "ol_subject"
    """Open Library shared-subject edge; a fallback, not a behavioural signal."""


class SourceName(StrEnum):
    """Which ingest adapter produced a row."""

    GOODREADS_UCSD = "goodreads_ucsd"
    AMAZON_META = "amazon_meta"
    AMAZON_REVIEWS_2023 = "amazon_reviews_2023"
    OPENLIBRARY = "openlibrary"
    DEMO = "demo"


@dataclass(frozen=True, slots=True)
class Book:
    """A work-level book record.

    ``work_id`` is bookmap's own canonical identifier, assigned by
    :mod:`bookmap.identity`. The upstream identifiers are kept alongside it so
    that edges from different sources can be resolved onto the same node, and
    so any node can be traced back to its origin.
    """

    work_id: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    isbn13: str | None = None
    asin: str | None = None
    goodreads_id: str | None = None
    openlibrary_id: str | None = None
    series: str | None = None
    avg_rating: float | None = None
    ratings_count: int | None = None
    subjects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.work_id:
            raise ValueError("work_id must be non-empty")
        if not self.title:
            raise ValueError(f"title must be non-empty (work_id={self.work_id!r})")


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed, typed, ranked connection between two works.

    ``rank`` is the 1-based position the target held in the source's list.
    Position carries real signal — the first "also bought" slot is a far
    stronger recommendation than the twentieth — so it is retained through
    ingest and only collapsed into a scalar weight during fusion
    (:func:`bookmap.graph.build.fuse_edges`).
    """

    src: str
    dst: str
    kind: EdgeKind
    rank: int = 1
    source: SourceName = SourceName.DEMO

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank is 1-based, got {self.rank}")
        if self.src == self.dst:
            raise ValueError(f"self-loop on {self.src!r}")


@dataclass(frozen=True, slots=True)
class FusedEdge:
    """An undirected edge after multi-source fusion.

    ``weight`` is the fused strength; ``dir_asym`` records how lopsided the
    original directed pair was (0.0 = perfectly mutual, 1.0 = one-way only),
    which is what lets the map distinguish "these two books recommend each
    other" from "a bestseller is listed under an obscure title".
    """

    src: str
    dst: str
    weight: float
    dir_asym: float = 0.0
    kinds: tuple[EdgeKind, ...] = ()


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A scored recommendation with its provenance in the graph."""

    work_id: str
    title: str
    authors: tuple[str, ...] = ()
    score: float = 0.0
    raw_ppr: float = 0.0
    degree: int = 0
    community: int | None = None
    explanations: tuple[Explanation, ...] = ()


@dataclass(frozen=True, slots=True)
class Explanation:
    """Why a book was recommended: a concrete path back to one of the seeds."""

    seed_id: str
    seed_title: str
    path: tuple[str, ...] = field(default=())
    """Work IDs from seed to recommendation inclusive; length 2 means adjacent."""

    strength: float = 0.0

    @property
    def hops(self) -> int:
        return max(len(self.path) - 1, 0)


@dataclass(frozen=True, slots=True)
class SeedMatch:
    """The result of resolving a user-typed book title onto a work.

    Ambiguity is surfaced rather than silently resolved: if a query matches
    several works closely, ``alternatives`` is populated so the caller (CLI or
    web UI) can ask instead of guessing.
    """

    query: str
    work_id: str | None
    title: str | None = None
    authors: tuple[str, ...] = ()
    confidence: float = 0.0
    alternatives: tuple[tuple[str, str], ...] = ()
    """(work_id, display label) pairs for near-ties."""

    @property
    def resolved(self) -> bool:
        return self.work_id is not None

    @property
    def ambiguous(self) -> bool:
        return len(self.alternatives) > 0
