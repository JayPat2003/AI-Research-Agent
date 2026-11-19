"""Request/response Pydantic models for the public HTTP API.

Kept separate from agent schemas so transport contracts can evolve without
changing LangGraph structured-output shapes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for ``POST /query`` — natural-language research question."""

    query: str
    conversation_id: str | None = None
    user_id: str | None = None


class IngestUrlRequest(BaseModel):
    """Body for ``POST /ingest/url``."""

    url: str
    domain: str | None = None
    topic: str | None = None


class IngestArxivRequest(BaseModel):
    """Body for ``POST /ingest/arxiv``."""

    arxiv_id: str
    domain: str | None = None


class IngestTextRequest(BaseModel):
    """Body for ``POST /ingest/text`` (pasted notes)."""

    text: str
    title: str = "Pasted note"
    domain: str | None = None


class PreferenceUpdate(BaseModel):
    """Partial update for ``PUT /preferences/{user_id}``."""

    domain: str | None = None
    citation_style: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
