"""The contract every ingest adapter implements.

Adapters are deliberately dumb: they yield ``Book`` and ``Edge`` records in
whatever order the upstream format makes cheap, and know nothing about
identity resolution, fusion, or storage. That keeps a new data source to one
small file.

Note on identifiers: adapters emit edges keyed by *upstream* identifiers
(Goodreads book ids, ASINs) using :meth:`Source.ref` to tag them. The store
resolves those onto canonical ``work_id`` values after all books are known,
because a source routinely references a book before it describes it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from bookmap.models import Book, Edge, SourceName


REF_ID_TYPE: dict[SourceName, str] = {
    SourceName.GOODREADS_UCSD: "goodreads_id",
    SourceName.AMAZON_META: "asin",
    SourceName.AMAZON_REVIEWS_2023: "asin",
    SourceName.OPENLIBRARY: "openlibrary_id",
    SourceName.DEMO: "demo_id",
}
"""Which ``aliases.id_type`` a source's namespaced refs resolve through.

Shared by the adapters (which emit refs) and the store (which rewrites them), so
the two cannot drift apart.
"""


@runtime_checkable
class Source(Protocol):
    """A data source that can yield books and the connections between them."""

    name: SourceName

    def iter_books(self) -> Iterator[Book]:
        """Yield every book the source describes.

        Implementations should stream rather than materialise: the Goodreads
        dump is ~2GB gzipped and will not fit in memory.
        """
        ...

    def iter_edges(self) -> Iterator[Edge]:
        """Yield every connection the source asserts, with 1-based ``rank``."""
        ...


def unresolved_ref(source: SourceName, raw_id: str) -> str:
    """Tag an upstream identifier so it can be resolved later.

    Adapters cannot know a book's canonical ``work_id`` while streaming, so
    edges are emitted against namespaced references (``goodreads_ucsd:12345``)
    which :mod:`bookmap.store.db` rewrites once the book table is populated.
    """
    if not raw_id:
        raise ValueError("raw_id must be non-empty")
    return f"{source.value}:{raw_id}"


def split_ref(ref: str) -> tuple[str, str]:
    """Inverse of :func:`unresolved_ref`. Returns ``(source, raw_id)``."""
    source, _, raw_id = ref.partition(":")
    if not raw_id:
        raise ValueError(f"not a namespaced ref: {ref!r}")
    return source, raw_id


class FileSource:
    """Convenience base for adapters that read a local dump file."""

    name: SourceName

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such dump file: {self.path}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.path)!r})"
