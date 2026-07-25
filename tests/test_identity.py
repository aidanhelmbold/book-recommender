"""Identifier canonicalisation and seed resolution.

The ISBN expectations are real check digits, computed by hand from the standard
algorithms -- not captured from the implementation.
"""

from __future__ import annotations

import pytest

from bookmap.identity import (
    canonical_isbn,
    isbn10_to_13,
    isbn13_check_digit,
    looks_like_asin,
    make_work_id,
    normalize_author,
    normalize_title,
    resolve_seed,
    valid_isbn13,
)


class TestNormalizeTitle:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Dune", "dune"),
            ("Dune (Dune Chronicles, #1)", "dune"),
            ("Dune: Book One", "dune"),
            ("  DUNE  ", "dune"),
            ("The Hobbit", "hobbit"),
            ("A Game of Thrones", "game of thrones"),
            ("An Absolutely Remarkable Thing", "absolutely remarkable thing"),
            ("Foundation and Empire", "foundation and empire"),
            ("Harry Potter and the Philosopher's Stone", "harry potter and the philosophers stone"),
            ("Slaughterhouse-Five", "slaughterhouse five"),
            ("The Left Hand of Darkness [Reissue]", "left hand of darkness"),
        ],
    )
    def test_known_normalizations(self, raw: str, expected: str) -> None:
        assert normalize_title(raw) == expected

    def test_editions_of_one_work_collapse(self) -> None:
        variants = [
            "Dune",
            "Dune (Dune Chronicles, #1)",
            "Dune: Book One",
            "  dune  ",
        ]
        assert len({normalize_title(v) for v in variants}) == 1

    def test_only_leading_articles_are_stripped(self) -> None:
        """Internal articles carry meaning; dropping them would merge distinct works."""
        assert normalize_title("The Lord of the Rings") == "lord of the rings"

    def test_distinct_works_stay_distinct(self) -> None:
        assert normalize_title("Foundation") != normalize_title("Foundation and Empire")

    def test_empty_input(self) -> None:
        assert normalize_title("") == ""


class TestNormalizeAuthor:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Frank Herbert", "frank herbert"),
            ("Herbert, Frank", "frank herbert"),
            ("  Frank   Herbert ", "frank herbert"),
            ("Ursula K. Le Guin", "ursula k le guin"),
            ("J.R.R. Tolkien", "jrr tolkien"),
        ],
    )
    def test_known_normalizations(self, raw: str, expected: str) -> None:
        assert normalize_author(raw) == expected

    def test_inverted_and_plain_forms_agree(self) -> None:
        assert normalize_author("Herbert, Frank") == normalize_author("Frank Herbert")


class TestISBN:
    def test_isbn13_check_digit_known_values(self) -> None:
        # Dune (Ace): 978-0-441-01359-3
        assert isbn13_check_digit("978044101359") == "3"
        # 978-1-56101-074-5
        assert isbn13_check_digit("978156101074") == "5"

    def test_isbn10_to_13_known_values(self) -> None:
        assert isbn10_to_13("0441013597") == "9780441013593"
        # Exercises the 'X' check-digit path.
        assert isbn10_to_13("156101074X") == "9781561010745"

    def test_isbn10_to_13_accepts_hyphens(self) -> None:
        assert isbn10_to_13("0-441-01359-7") == "9780441013593"

    def test_isbn10_with_bad_check_digit_is_rejected(self) -> None:
        """Must return None, not a plausible-looking ISBN-13.

        Silently converting a corrupt identifier is worse than failing: it
        fabricates a valid-looking key that will merge two unrelated books.
        """
        assert isbn10_to_13("0441013598") is None

    @pytest.mark.parametrize("bad", ["", "12345", "not-an-isbn", "04410135970"])
    def test_isbn10_to_13_rejects_malformed(self, bad: str) -> None:
        assert isbn10_to_13(bad) is None

    def test_valid_isbn13(self) -> None:
        assert valid_isbn13("9780441013593")
        assert valid_isbn13("978-0-441-01359-3")
        assert not valid_isbn13("9780441013594")
        assert not valid_isbn13("")
        assert not valid_isbn13("0441013597")

    def test_canonical_isbn_accepts_both_widths(self) -> None:
        assert canonical_isbn("0441013597") == "9780441013593"
        assert canonical_isbn("9780441013593") == "9780441013593"
        assert canonical_isbn("978-0-441-01359-3") == "9780441013593"

    def test_canonical_isbn_rejects_junk(self) -> None:
        for bad in (None, "", "0000000000", "B000FC1PWA"):
            assert canonical_isbn(bad) is None


class TestLooksLikeASIN:
    def test_amazon_assigned_asin(self) -> None:
        assert looks_like_asin("B000FC1PWA")

    def test_isbn10_reused_as_asin_is_not_an_asin(self) -> None:
        """Amazon reuses ISBN-10s as ASINs for books.

        Treating those as opaque ASINs would prevent them joining to the
        Goodreads graph via ISBN, which is the whole point of resolution.
        """
        assert not looks_like_asin("0441013597")

    @pytest.mark.parametrize("bad", [None, "", "B000", "TOOLONGVALUE12"])
    def test_rejects_malformed(self, bad: str | None) -> None:
        assert not looks_like_asin(bad)


class TestMakeWorkId:
    def test_is_deterministic(self) -> None:
        assert make_work_id("Dune", ["Frank Herbert"]) == make_work_id("Dune", ["Frank Herbert"])

    def test_collapses_edition_decoration_and_author_form(self) -> None:
        """The load-bearing test for cross-source joins.

        Goodreads says "Dune (Dune Chronicles, #1)" by "Frank Herbert"; another
        source says "Dune" by "Herbert, Frank". These must land on one node or
        the fused graph splits into per-source components.
        """
        assert make_work_id("Dune (Dune Chronicles, #1)", ["Frank Herbert"]) == make_work_id(
            "Dune", ["Herbert, Frank"]
        )

    def test_distinct_works_get_distinct_ids(self) -> None:
        assert make_work_id("Dune", ["Frank Herbert"]) != make_work_id(
            "Dune Messiah", ["Frank Herbert"]
        )

    def test_same_title_different_author_is_a_different_work(self) -> None:
        assert make_work_id("Foundation", ["Isaac Asimov"]) != make_work_id(
            "Foundation", ["Mercedes Lackey"]
        )

    def test_only_primary_author_matters(self) -> None:
        """Co-author lists vary between sources; the first author is the stable part."""
        assert make_work_id("Good Omens", ["Terry Pratchett", "Neil Gaiman"]) == make_work_id(
            "Good Omens", ["Terry Pratchett"]
        )

    def test_missing_author_still_yields_an_id(self) -> None:
        work_id = make_work_id("Beowulf", [])
        assert isinstance(work_id, str)
        assert work_id


CANDIDATES = [
    ("w-dune", "Dune", ("Frank Herbert",)),
    ("w-dune-messiah", "Dune Messiah", ("Frank Herbert",)),
    ("w-foundation", "Foundation", ("Isaac Asimov",)),
    ("w-foundation-empire", "Foundation and Empire", ("Isaac Asimov",)),
    ("w-hyperion", "Hyperion", ("Dan Simmons",)),
]


class TestResolveSeed:
    def test_exact_title_match(self) -> None:
        match = resolve_seed("Dune", CANDIDATES)
        assert match.work_id == "w-dune"
        assert match.resolved

    def test_case_and_whitespace_insensitive(self) -> None:
        assert resolve_seed("  dUnE ", CANDIDATES).work_id == "w-dune"

    def test_tolerates_a_typo(self) -> None:
        assert resolve_seed("Hyperian", CANDIDATES).work_id == "w-hyperion"

    def test_author_in_query_disambiguates(self) -> None:
        match = resolve_seed("foundation asimov", CANDIDATES)
        assert match.work_id == "w-foundation"

    def test_nonsense_query_is_unresolved(self) -> None:
        match = resolve_seed("zzzzz not a real book zzzzz", CANDIDATES)
        assert not match.resolved
        assert match.work_id is None

    def test_near_ties_are_reported_not_guessed(self) -> None:
        """Two editions scoring within the margin must both be offered.

        Silently picking one is how a user ends up with recommendations for a
        book they did not mean.
        """
        candidates = [
            ("w-a", "The Dispossessed", ("Ursula K. Le Guin",)),
            ("w-b", "The Dispossessed", ("Ursula Le Guin",)),
        ]
        match = resolve_seed("The Dispossessed", candidates)
        assert match.ambiguous
        assert len(match.alternatives) >= 2

    def test_unambiguous_match_has_no_alternatives(self) -> None:
        assert not resolve_seed("Hyperion", CANDIDATES).ambiguous

    def test_confidence_is_higher_for_exact_matches(self) -> None:
        exact = resolve_seed("Dune", CANDIDATES)
        fuzzy = resolve_seed("Hyperian", CANDIDATES)
        assert exact.confidence > fuzzy.confidence
        assert 0.0 <= fuzzy.confidence <= 1.0

    def test_threshold_is_respected(self) -> None:
        assert not resolve_seed("Hyperian", CANDIDATES, threshold=0.99).resolved

    def test_query_is_echoed_back(self) -> None:
        assert resolve_seed("Dune", CANDIDATES).query == "Dune"

    def test_empty_candidate_list(self) -> None:
        assert not resolve_seed("Dune", []).resolved


class TestTieBreakPrefersCallerOrder:
    """Ties must break toward the caller's ordering, not toward a hash.

    Found on the real dump. ``normalize_title`` truncates at a subtitle marker, so
    "Dune - The Official Comic Book" and "Foundation: Redefine Your Core, Conquer
    Back Pain" normalise to exactly "dune" and "foundation" — tying with the books
    anyone actually means. The store hands candidates over most-rated first, but
    resolve_seed re-sorted on work_id and discarded that, so seeding "Foundation"
    resolved to a fitness book and returned twenty pages of exercise manuals.

    Ranking the candidates is the caller's job — it has the ratings counts. All
    this needs to do is not destroy that order.
    """

    # Most-rated first, as the store returns them.
    BY_POPULARITY = [
        ("w-real-dune", "Dune", ("Frank Herbert",)),
        ("w-comic", "Dune - The Official Comic Book", ("Bill Sienkiewicz",)),
        ("w-gateway", "Dune: The Gateway Collection", ("Frank Herbert",)),
    ]

    def test_first_candidate_wins_an_exact_tie(self) -> None:
        assert resolve_seed("Dune", self.BY_POPULARITY).work_id == "w-real-dune"

    def test_holds_whatever_the_work_ids_are(self) -> None:
        """A work id that sorts first alphabetically must not win on that account."""
        candidates = [
            ("zzz-real", "Foundation", ("Isaac Asimov",)),
            ("aaa-fitness", "Foundation: Redefine Your Core, Conquer Back Pain", ("Eric Goodman",)),
        ]
        assert resolve_seed("Foundation", candidates).work_id == "zzz-real"

    def test_still_reports_the_tie_as_ambiguous(self) -> None:
        """Picking a sensible default does not mean hiding that it was a guess."""
        match = resolve_seed("Dune", self.BY_POPULARITY)
        assert match.ambiguous
