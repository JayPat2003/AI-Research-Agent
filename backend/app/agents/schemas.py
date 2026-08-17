from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    intent: Literal["knowledge", "research", "comparison", "data", "calculation", "web"]
    rewritten_query: str
    needs_retrieval: bool = True
    needs_web: bool = False
    needs_sql: bool = False
    rationale: str = ""


class SQLDecision(BaseModel):
    sql: str
    explanation: str


class Citation(BaseModel):
    chunk_id: str | None = None
    title: str
    url: str | None = None
    source: str = ""
    note: str = ""


class CitedReport(BaseModel):
    title: str
    executive_summary: str
    body_markdown: str
    recommendation: str
    limitations: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    pass_check: bool
    claims_supported: bool
    citations_present: bool
    sources_relevant: bool
    contradictions: bool
    hallucinating: bool
    answered_question: bool
    issues: list[str] = Field(default_factory=list)
    missing_queries: list[str] = Field(default_factory=list)
