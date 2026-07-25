"""``bookmap`` command line interface.

    bookmap ingest demo
    bookmap ingest goodreads-ucsd ~/Downloads/goodreads_books.json.gz
    bookmap build --min-weight 0.05
    bookmap recommend --seeds "Dune,Foundation,Hyperion" -n 20
    bookmap map --seeds "Dune,Foundation" --out map.html
    bookmap web
    bookmap stats
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="bookmap",
    help="Build and query a book recommendation network.",
    no_args_is_help=True,
)

ingest_app = typer.Typer(help="Load a data source into the graph store.")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("demo")
def ingest_demo(db: str = "bookmap.duckdb", path: str | None = None) -> None:
    """Load the bundled hand-authored demo corpus."""
    raise NotImplementedError


@ingest_app.command("goodreads-ucsd")
def ingest_goodreads(path: str, db: str = "bookmap.duckdb", max_rank: int = 50) -> None:
    """Load goodreads_books.json.gz from the UCSD Book Graph."""
    raise NotImplementedError


@ingest_app.command("amazon-meta")
def ingest_amazon_meta(path: str, db: str = "bookmap.duckdb", max_rank: int = 50) -> None:
    """Load SNAP amazon-meta.txt co-purchase data."""
    raise NotImplementedError


@ingest_app.command("amazon-reviews")
def ingest_amazon_reviews(path: str, db: str = "bookmap.duckdb", max_rank: int = 50) -> None:
    """Load Amazon Reviews 2023 book metadata (also_buy / also_view)."""
    raise NotImplementedError


@ingest_app.command("openlibrary")
def ingest_openlibrary(db: str = "bookmap.duckdb", limit: int | None = None) -> None:
    """Enrich stored books with Open Library metadata."""
    raise NotImplementedError


@app.command()
def build(
    db: str = "bookmap.duckdb",
    min_weight: float = 0.05,
    drop_leaves: bool = False,
    max_rank: int = 50,
    communities: bool = True,
    betweenness: bool = False,
) -> None:
    """Fuse raw edges into the weighted graph and compute node metrics."""
    raise NotImplementedError


@app.command()
def recommend(
    seeds: str = typer.Option(..., help="Comma-separated book titles."),
    n: int = 20,
    db: str = "bookmap.duckdb",
    beta: float = 0.25,
    mmr_lambda: float = 0.7,
    exclude_same_author: bool = False,
    explain: bool = True,
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Recommend books similar to a set of seed titles."""
    raise NotImplementedError


@app.command("map")
def map_cmd(
    seeds: str = typer.Option(..., help="Comma-separated book titles."),
    out: str = "map.html",
    n: int = 40,
    db: str = "bookmap.duckdb",
) -> None:
    """Export a standalone HTML map of the seeds' neighbourhood."""
    raise NotImplementedError


@app.command()
def web(db: str = "bookmap.duckdb", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the interactive map."""
    raise NotImplementedError


@app.command()
def stats(db: str = "bookmap.duckdb") -> None:
    """Print graph size, degree distribution, and community summary."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    app()
