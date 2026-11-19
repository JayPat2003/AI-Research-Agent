"""Retrieval DTOs and the vector-store Protocol.

``RetrievedChunk`` is the canonical hit shape used by the graph, API, and eval.
``VectorStore`` abstracts the physical index so pgvector today can be swapped
for Qdrant (or similar) without rewriting :class:`~app.retrieval.service.RetrievalService`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RetrievedChunk:
    """One ranked evidence passage from the private knowledge base."""

    chunk_id: str
    document_id: str
    content: str
    title: str
    source: str
    url: str | None
    page_number: int | None
    score: float
    rank: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def citation_label(self) -> str:
        """Short human-readable citation string for prompts / UI."""
        page = f", p.{self.page_number}" if self.page_number else ""
        return f"{self.title}{page} [{self.chunk_id[:8]}]"


@dataclass
class MetadataFilter:
    """Optional document-level filters applied during search."""

    domain: str | None = None
    source: str | None = None
    topic: str | None = None
    author: str | None = None


@runtime_checkable
class VectorStore(Protocol):
    """Retrieval backend. pgvector is Phase 1; Qdrant can implement this later."""

    async def dense_search(
        self, query_embedding: list[float], k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]: ...

    async def sparse_search(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]: ...
