"""Identifier canonicalisation and fuzzy title resolution.

Two jobs, both load-bearing:

1. Cross-source identity. Amazon speaks ASINs, Goodreads speaks its own ids,
   libraries speak ISBNs. Without collapsing these onto one work, the fused
   graph is two disconnected components that happen to describe the same books.
2. Seed resolution. The user types "Dune", not a work id.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence

from rapidfuzz import fuzz

from bookmap.models import SeedMatch

_STOPWORDS = frozenset({"a", "an", "the"})

_BRACKETED = re.compile(r"[(\[][^)\]]*[)\]]?")
"""Parenthesised or bracketed decoration: "(Dune Chronicles, #1)", "[Reissue]".
The closing bracket is optional because dump titles are routinely truncated."""

_SUBTITLE = re.compile(r"\s*(?::|\s[-–—]\s|--)")
"""A subtitle/series marker. A dash only counts when it is spaced or doubled, so
"Slaughterhouse-Five" keeps its second half."""

_ELIDED = re.compile(r"['’ʼ]")
"""Apostrophes vanish rather than becoming spaces: "philosopher's" is one word."""

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_AUTHOR_BONUS = 0.15
"""How much of the remaining distance to a perfect score an author match closes.
Naming the author is corroboration, not proof, so it may not outrank a better
title match on its own."""

_MIN_AUTHOR_TOKEN = 3
"""Shorter author tokens ("k.", "le", "jr") match too many queries by accident."""


def normalize_title(title: str) -> str:
    """Reduce a title to a comparable key.

    Lowercases, strips punctuation and leading articles, collapses whitespace,
    and drops parenthesised/bracketed suffixes and any trailing series marker
    after a colon or dash. Editions differ mostly in exactly this decoration --
    "Dune (Dune Chronicles, #1)" and "Dune: Book One" must both key to "dune".
    """
    if not title:
        return ""

    # Fold accents so "Les Misérables" and "Les Miserables" agree.
    text = unicodedata.normalize("NFKD", title.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _BRACKETED.sub(" ", text)

    # Truncate at the first subtitle marker, but only if something survives:
    # a title that is *only* a subtitle would otherwise normalise to nothing.
    head = _SUBTITLE.split(text, maxsplit=1)[0]
    if head.strip():
        text = head

    text = _NON_ALNUM.sub(" ", _ELIDED.sub("", text))
    words = text.split()
    # Only a *leading* article is decoration; internal ones carry meaning, so
    # "The Lord of the Rings" and "Lord of Rings" must not collapse together.
    if words and words[0] in _STOPWORDS:
        words = words[1:]
    return " ".join(words)


def normalize_author(author: str) -> str:
    """Reduce an author name to a comparable key ("Herbert, Frank" -> "frank herbert")."""
    if not author:
        return ""

    text = unicodedata.normalize("NFKD", author.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Library dumps invert names; Goodreads does not. Un-invert on the first
    # comma so the two forms produce the same key.
    surname, comma, forenames = text.partition(",")
    if comma:
        text = f"{forenames} {surname}"

    # Initials lose their dots without gaining a space, so "J.R.R." keys as
    # "jrr" and matches a source that wrote "JRR".
    text = _NON_ALNUM.sub(" ", text.replace(".", ""))
    return " ".join(text.split())


def _isbn_digits(raw: str, width: int) -> str | None:
    """Strip separators and validate the shape of an ISBN of ``width`` digits.

    Returns None unless the result is exactly ``width`` characters of digits
    (with a trailing 'X' allowed for ISBN-10), or is the all-zero placeholder
    that dumps use to mean "unknown" -- accepting that would merge every book
    whose ISBN was missing onto one identifier.
    """
    text = re.sub(r"[\s-]", "", raw).upper()
    if len(text) != width:
        return None
    body, check = text[:-1], text[-1]
    if not body.isdigit():
        return None
    if not (check.isdigit() or (width == 10 and check == "X")):
        return None
    if set(body) == {"0"}:
        return None
    return text


def isbn10_to_13(isbn10: str) -> str | None:
    """Convert an ISBN-10 to ISBN-13, or return None if it is not a valid ISBN-10.

    Validates the ISBN-10 check digit before converting; an invalid input is
    rejected rather than silently producing a plausible-looking ISBN-13.
    """
    if not isbn10:
        return None
    text = _isbn_digits(isbn10, 10)
    if text is None:
        return None

    # ISBN-10 checksum: sum of digit * (10 - position) is divisible by 11, with
    # 'X' standing for 10.
    total = sum(
        (10 if ch == "X" else int(ch)) * (10 - position) for position, ch in enumerate(text)
    )
    if total % 11 != 0:
        return None

    first12 = f"978{text[:9]}"
    return first12 + isbn13_check_digit(first12)


def isbn13_check_digit(first12: str) -> str:
    """Return the ISBN-13 check digit for the first 12 digits."""
    digits = re.sub(r"[\s-]", "", first12)
    if len(digits) != 12 or not digits.isdigit():
        raise ValueError(f"expected 12 digits, got {first12!r}")
    total = sum(int(ch) * (3 if position % 2 else 1) for position, ch in enumerate(digits))
    return str(-total % 10)


def valid_isbn13(isbn13: str) -> bool:
    """True if the string is a well-formed ISBN-13 with a correct check digit."""
    if not isbn13:
        return False
    text = _isbn_digits(isbn13, 13)
    if text is None:
        return False
    return isbn13_check_digit(text[:12]) == text[12]


def canonical_isbn(raw: str | None) -> str | None:
    """Normalise any ISBN-ish string (hyphens, ISBN-10, X check digit) to ISBN-13."""
    if not raw:
        return None
    text = re.sub(r"[\s-]", "", raw).upper()
    if len(text) == 13:
        return text if valid_isbn13(text) else None
    if len(text) == 10:
        return isbn10_to_13(text)
    return None


def looks_like_asin(raw: str | None) -> bool:
    """True for a 10-character Amazon ASIN that is not simply an ISBN-10.

    Amazon reuses ISBN-10s as ASINs for books, so an all-digit value is treated
    as an ISBN; a leading 'B' marks a true Amazon-assigned ASIN.
    """
    if not raw:
        return False
    text = raw.strip().upper()
    if len(text) != 10 or not text.isalnum():
        return False
    # An ISBN-10 shape belongs to the ISBN namespace, where it can still join to
    # the Goodreads graph; calling it an opaque ASIN would strand it.
    return re.fullmatch(r"\d{9}[\dX]", text) is None


def make_work_id(title: str, authors: Sequence[str]) -> str:
    """Derive a deterministic canonical work id from title and primary author.

    Deterministic so that re-ingesting the same dump, or ingesting two sources
    that describe the same book, yields the same node rather than a duplicate.
    """
    title_key = normalize_title(title)
    # Only the primary author: co-author lists differ between sources (and are
    # often reordered), so including them all would split one work in two.
    author_key = normalize_author(authors[0]) if authors else ""

    # The hash carries the identity; the slug is there so a work id read out of
    # a log or a URL is recognisable.
    digest = hashlib.blake2b(f"{title_key}|{author_key}".encode(), digest_size=8).hexdigest()
    slug = title_key.replace(" ", "-")[:48].strip("-")
    return f"w-{slug}-{digest}" if slug else f"w-{digest}"


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
    query_norm = normalize_title(query)
    query_tokens = query_norm.split()

    scored: list[tuple[float, str, str, tuple[str, ...]]] = []
    for position, (work_id, title, authors) in enumerate(candidates):
        author_tokens = {
            token
            for author in authors
            for token in normalize_author(author).split()
            if len(token) >= _MIN_AUTHOR_TOKEN
        }
        named = author_tokens.intersection(query_tokens)

        # An author named in the query must not then be held against the title.
        # Scoring "foundation asimov" against "Foundation" beats scoring it
        # against a query the title could never contain in full.
        remainder = " ".join(t for t in query_tokens if t not in named) or query_norm

        # WRatio rather than a plain edit ratio: a user typing part of a long
        # title ("harry potter") should still find it.
        base = fuzz.WRatio(remainder, normalize_title(title)) / 100.0
        score = base + (1.0 - base) * _AUTHOR_BONUS if named else base
        scored.append((score, position, work_id, title, tuple(authors)))

    if not scored:
        return SeedMatch(query=query, work_id=None)

    # Tie-break on the caller's ordering, not on work_id. Callers rank candidates
    # by ratings count, and many real titles normalise identically -- "Dune" and
    # "Dune - The Official Comic Book" both key to "dune" -- so sorting on an
    # opaque id here silently picked whichever book happened to hash lowest.
    scored.sort(key=lambda row: (-row[0], row[1]))
    best_score, _, best_id, best_title, best_authors = scored[0]

    if best_score < threshold:
        # Report how close it got: a caller can show "did you mean" without
        # having to re-run the search itself.
        return SeedMatch(query=query, work_id=None, confidence=best_score)

    near = [row for row in scored[:limit] if best_score - row[0] <= ambiguity_margin]
    alternatives = (
        tuple((work_id, _label(title, authors)) for _, _, work_id, title, authors in near)
        if len(near) > 1
        else ()
    )

    return SeedMatch(
        query=query,
        work_id=best_id,
        title=best_title,
        authors=best_authors,
        confidence=min(best_score, 1.0),
        alternatives=alternatives,
    )


def _label(title: str, authors: Sequence[str]) -> str:
    """A human-readable disambiguation label; the caller shows this verbatim."""
    return f"{title} — {authors[0]}" if authors else title
