from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    user_id: str | None = None


class IngestUrlRequest(BaseModel):
    url: str
    domain: str | None = None
    topic: str | None = None


class IngestArxivRequest(BaseModel):
    arxiv_id: str
    domain: str | None = None


class IngestTextRequest(BaseModel):
    text: str
    title: str = "Pasted note"
    domain: str | None = None


class PreferenceUpdate(BaseModel):
    domain: str | None = None
    citation_style: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
