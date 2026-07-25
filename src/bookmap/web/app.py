"""FastAPI app serving the interactive book map.

The store is opened read-only so serving can never contend with an ingest.
Layout is computed per request over the induced subgraph around the seeds and
results; the full graph is never laid out.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    seeds: list[str] = Field(..., min_length=1)
    n: int = Field(20, ge=1, le=200)
    beta: float = Field(0.25, ge=0.0, le=1.0)
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    exclude_same_author: bool = False


class SubgraphRequest(BaseModel):
    seeds: list[str] = Field(..., min_length=1)
    n: int = Field(40, ge=1, le=400)
    include_neighbors: bool = True


def create_app(db_path: str = "bookmap.duckdb") -> FastAPI:
    """Build the app.

    Endpoints:
      ``GET  /``                       the map UI
      ``GET  /api/search?q=``          typeahead seed resolution
      ``POST /api/recommend``          ranked recommendations with explanations
      ``POST /api/subgraph``           nodes with layout positions, edges, communities
      ``GET  /api/book/{work_id}``     one book's details
      ``GET  /api/book/{id}/neighbors``click-to-expand
      ``GET  /api/stats``              graph summary

    Unknown seeds return 200 with them listed under ``unresolved`` rather than
    an error: a partially-resolvable seed set is still useful.
    """
    raise NotImplementedError


def build_subgraph_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Assemble the node/edge/position JSON the canvas renderer consumes."""
    raise NotImplementedError
