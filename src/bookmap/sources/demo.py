"""Bundled demo corpus loader.

The demo corpus exists so the whole pipeline is runnable and testable without a
multi-gigabyte download. It is hand-authored, and labelled as such in the data
file itself: it is *not* scraped Amazon or Goodreads output and must never be
presented as such. It is small enough to reason about and shaped like the real
thing -- several genre clusters, a few deliberate bridge books, a realistic
degree skew.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bookmap.models import Book, Edge, EdgeKind, SourceName
from bookmap.sources.base import unresolved_ref

DEFAULT_DEMO_PATH = Path(__file__).resolve().parents[3] / "data" / "demo" / "corpus.json"


class DemoSource:
    """Loads the bundled JSON corpus."""

    name = SourceName.DEMO

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DEMO_PATH
        self._payload: dict[str, Any] | None = None
        if path is not None and not self.path.exists():
            # An explicit path is caller input, so it is checked immediately
            # where the caller can still fix the typo. The bundled default is
            # checked on load instead: a missing one is a packaging fault, and
            # the path must remain inspectable to report it.
            raise FileNotFoundError(f"no such demo corpus: {self.path}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.path)!r})"

    def _load(self) -> dict[str, Any]:
        """Read and cache the corpus.

        Cached because it is small by construction and both iterators need it;
        unlike the real dumps there is nothing to stream.
        """
        if self._payload is None:
            with self.path.open("r", encoding="utf-8") as handle:
                self._payload = json.load(handle)
        return self._payload

    def iter_books(self) -> Iterator[Book]:
        for record in self._load().get("books") or ():
            book_id = str(record.get("id") or "").strip()
            title = str(record.get("title") or "").strip()
            if not book_id or not title:
                continue
            yield Book(
                # The corpus id *is* the work id: this data has no upstream
                # identifiers to reconcile, so inventing a second namespace for
                # it would only add a mapping that can go wrong.
                work_id=book_id,
                title=title,
                authors=tuple(
                    author
                    for raw in record.get("authors") or ()
                    if (author := str(raw).strip())
                ),
                year=record.get("year"),
                subjects=tuple(
                    subject
                    for raw in record.get("subjects") or ()
                    if (subject := str(raw).strip())
                ),
            )

    def iter_edges(self) -> Iterator[Edge]:
        """Yield the corpus' declared edges, ranked by list position."""
        for record in self._load().get("edges") or ():
            src_id = str(record.get("src") or "").strip()
            if not src_id:
                continue
            # An unknown kind raises rather than defaulting: the corpus is
            # hand-authored, so a typo there is a mistake to surface, not data
            # to absorb.
            kind = EdgeKind(record.get("kind") or EdgeKind.GR_SIMILAR)
            src = unresolved_ref(self.name, src_id)
            for position, raw_target in enumerate(record.get("similar") or (), start=1):
                target = str(raw_target or "").strip()
                if not target or target == src_id:
                    continue
                yield Edge(
                    src=src,
                    dst=unresolved_ref(self.name, target),
                    kind=kind,
                    rank=position,
                    # Provenance is carried explicitly so a demo edge can never
                    # be read as real co-purchase evidence downstream.
                    source=SourceName.DEMO,
                )
