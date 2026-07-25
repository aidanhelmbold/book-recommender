"""Identifier canonicalisation and fuzzy title resolution.

Two jobs, both load-bearing:

1. Cross-source identity. Amazon speaks ASINs, Goodreads speaks its own ids,
   libraries speak ISBNs. Without collapsing these onto one work, the fused
   graph is two disconnected components that happen to describe the same books.
2. Seed resolution. The user types "Dune", not a work id.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from bookmap.models import SeedMatch

_STOPWORDS = frozenset({"a", "an", "the"})


def normalize_title(title: str) -> str:
    """Reduce a title to a comparable key.

    Lowercases, strips punctuation and leading articles, collapses whitespace,
    and drops parenthesised/bracketed suffixes and any trailing series marker
    after a colon or dash. Editions differ mostly in exactly this decoration --
    "Dune (Dune Chronicles, #1)" and "Dune: Book One" must both key to "dune".
    """
    raise NotImplementedError


def normalize_author(author: str) -> str:
    """Reduce an author name to a comparable key ("Herbert, Frank" -> "frank herbert")."""
    raise NotImplementedError


def isbn10_to_13(isbn10: str) -> str | None:
    """Convert an ISBN-10 to ISBN-13, or return None if it is not a valid ISBN-10.

    Validates the ISBN-10 check digit before converting; an invalid input is
    rejected rather than silently producing a plausible-looking ISBN-13.
    """
    raise NotImplementedError


def isbn13_check_digit(first12: str) -> str:
    """Return the ISBN-13 check digit for the first 12 digits."""
    raise NotImplementedError


def valid_isbn13(isbn13: str) -> bool:
    """True if the string is a well-formed ISBN-13 with a correct check digit."""
    raise NotImplementedError


def canonical_isbn(raw: str | None) -> str | None:
    """Normalise any ISBN-ish string (hyphens, ISBN-10, X check digit) to ISBN-13."""
    raise NotImplementedError


def looks_like_asin(raw: str | None) -> bool:
    """True for a 10-character Amazon ASIN that is not simply an ISBN-10.

    Amazon reuses ISBN-10s as ASINs for books, so an all-digit value is treated
    as an ISBN; a leading 'B' marks a true Amazon-assigned ASIN.
    """
    raise NotImplementedError


def make_work_id(title: str, authors: Sequence[str]) -> str:
    """Derive a deterministic canonical work id from title and primary author.

    Deterministic so that re-ingesting the same dump, or ingesting two sources
    that describe the same book, yields the same node rather than a duplicate.
    """
    raise NotImplementedError


def resolve_seed(
    query: str,
    candidates: Iterable[tuple[str, str, tuple[str, ...]]],
    *,
    limit: int = 5,
    threshold: float = 0.75,
    ambiguity_margin: float = 0.05,
) -> SeedMatch:
    """Resolve a user-typed title onto a work.

    ``candidates`` yields ``(work_id, title, authors)``. A query may name an
    author too ("dune herbert"), which should raise the score of a matching
    candidate.

    Near-ties are reported rather than resolved: if the runner-up scores within
    ``ambiguity_margin`` of the winner, both land in
    :attr:`SeedMatch.alternatives` so the caller can disambiguate. Nothing above
    ``threshold`` yields an unresolved match.
    """
    raise NotImplementedError
