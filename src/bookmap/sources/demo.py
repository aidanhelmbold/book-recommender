"""Bundled demo corpus loader.

The demo corpus exists so the whole pipeline is runnable and testable without a
multi-gigabyte download. It is hand-authored, and labelled as such in the data
file itself: it is *not* scraped Amazon or Goodreads output and must never be
presented as such. It is small enough to reason about and shaped like the real
thing -- several genre clusters, a few deliberate bridge books, a realistic
degree skew.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from bookmap.models import Book, Edge, SourceName

DEFAULT_DEMO_PATH = Path(__file__).resolve().parents[3] / "data" / "demo" / "corpus.json"


class DemoSource:
    """Loads the bundled JSON corpus."""

    name = SourceName.DEMO

    def __init__(self, path: str | Path | None = None) -> None:
        raise NotImplementedError

    def iter_books(self) -> Iterator[Book]:
        raise NotImplementedError

    def iter_edges(self) -> Iterator[Edge]:
        """Yield the corpus' declared edges, ranked by list position."""
        raise NotImplementedError
