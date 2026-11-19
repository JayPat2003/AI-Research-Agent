"""Structured LLM output contracts for agent nodes.

Pydantic models here are bound via ``with_structured_output`` (or JSON-parse
fallback in :mod:`app.agents.invoke`). Keeping schemas separate from prompts
and graph logic makes contracts easy to version and unit-test.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Router output: intent classification and specialist tool flags."""

    intent: Literal["knowledge", "research", "comparison", "data", "calculation", "web"]
    rewritten_query: str
    needs_retrieval: bool = True
    needs_web: bool = False
    needs_sql: bool = False
    rationale: str = ""


class SQLDecision(BaseModel):
    """Text-to-SQL node output: a single SELECT plus human-readable rationale."""

    sql: str
    explanation: str


class CatalogFilter(BaseModel):
    """Legacy catalog-filter shape retained for eval / CLI compatibility."""

    drug_class: str | None = None
    condition: str | None = None
    status: str | None = None
    primary_endpoint: str | None = None
    explanation: str = ""


class Citation(BaseModel):
    """One source reference attached to a cited report."""

    chunk_id: str | None = None
    title: str
    url: str | None = None
    source: str = ""
    note: str = ""


class CitedReport(BaseModel):
    """Draft-agent product: executive briefing with citations and limitations."""

    title: str
    executive_summary: str
    body_markdown: str
    recommendation: str
    limitations: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    """Critic-agent verification checklist over a draft report."""

    pass_check: bool
    claims_supported: bool
    citations_present: bool
    sources_relevant: bool
    contradictions: bool
    hallucinating: bool
    answered_question: bool
    issues: list[str] = Field(default_factory=list)
    missing_queries: list[str] = Field(default_factory=list)
