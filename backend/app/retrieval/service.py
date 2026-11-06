from __future__ import annotations

import re
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.llm.embeddings import get_embedder, get_reranker
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.interface import MetadataFilter, RetrievedChunk
from app.retrieval.pgvector_store import PgVectorStore

Variant = Literal["naive", "hybrid", "hybrid_rerank"]

_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "vs",
    "versus",
    "about",
    "what",
    "which",
    "how",
    "is",
    "are",
    "do",
    "does",
    "adults",
    "patients",
    "treatment",
    "therapy",
}

# Too generic to count as on-topic evidence by themselves inside a medical corpus.
_WEAK_TOPIC = {
    "heart",
    "cardiac",
    "cardiovascular",
    "disease",
    "clinical",
    "study",
    "trial",
    "trials",
    "outcomes",
    "outcome",
    "patient",
    "medical",
    "health",
    "care",
    "risk",
    "blood",
    "drug",
    "drugs",
    "effect",
    "effects",
    "guidelines",
    "management",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOP}


def _topical_overlap(query: str, chunk: RetrievedChunk) -> float:
    q = _tokens(query)
    if not q:
        return 1.0
    doc = _tokens(f"{chunk.title} {chunk.content[:1200]}")
    if not doc:
        return 0.0
    distinctive = q - _WEAK_TOPIC
    focus = distinctive or q
    return len(focus & doc) / len(focus)


class RetrievalService:
    def __init__(self, session: AsyncSession) -> None:
        self.store = PgVectorStore(session)
        self.settings = get_settings()

    async def semantic_search(
        self, query: str, k: int | None = None, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        k = k or self.settings.dense_k
        embedding = get_embedder().embed_query(query)
        return await self.store.dense_search(embedding, k, filters)

    async def keyword_search(
        self, query: str, k: int | None = None, filters: MetadataFilter | None = None
    ) -> list[RetrievedChunk]:
        k = k or self.settings.sparse_k
        return await self.store.sparse_search(query, k, filters)

    async def hybrid_search(
        self,
        query: str,
        *,
        variant: Variant = "hybrid_rerank",
        filters: MetadataFilter | None = None,
        k: int | None = None,
    ) -> list[RetrievedChunk]:
        dense_k = self.settings.dense_k
        sparse_k = self.settings.sparse_k
        final_k = k or self.settings.rerank_k

        dense = await self.semantic_search(query, dense_k, filters)
        if variant == "naive":
            return self._keep_on_topic(query, dense[:final_k])

        sparse = await self.keyword_search(query, sparse_k, filters)
        fused = reciprocal_rank_fusion([dense, sparse])
        if variant == "hybrid":
            return self._keep_on_topic(query, fused[:final_k])

        return self.rerank(query, fused, final_k)

    def _keep_on_topic(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop off-topic seed hits that only match the medical domain vaguely."""
        floor = self.settings.retrieval_min_score
        kept: list[RetrievedChunk] = []
        for c in chunks:
            if float(c.score) < floor:
                continue
            overlap = _topical_overlap(query, c)
            # Require some query-term overlap unless the model score is clearly strong.
            if overlap < 0.15 and float(c.score) < max(floor, 0.35):
                continue
            kept.append(c)
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content,
                title=c.title,
                source=c.source,
                url=c.url,
                page_number=c.page_number,
                score=c.score,
                rank=rank,
                extra=c.extra,
            )
            for rank, c in enumerate(kept, start=1)
        ]

    def rerank(self, query: str, chunks: list[RetrievedChunk], k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.rerank_k
        if not chunks:
            return []
        scores = get_reranker().score(query, [c.content for c in chunks])
        paired = list(zip(chunks, scores, strict=False))
        paired.sort(key=lambda item: item[1], reverse=True)
        out: list[RetrievedChunk] = []
        floor = self.settings.retrieval_min_score
        for chunk, score in paired[: max(k * 2, k)]:
            if float(score) < floor:
                continue
            overlap = _topical_overlap(query, chunk)
            # Cross-encoder can still rank vaguely related medical text; require overlap or a clear score.
            if overlap < 0.15 and float(score) < max(floor, 0.25):
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    title=chunk.title,
                    source=chunk.source,
                    url=chunk.url,
                    page_number=chunk.page_number,
                    score=float(score),
                    rank=0,
                    extra=chunk.extra,
                )
            )
            if len(out) >= k:
                break
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content=c.content,
                title=c.title,
                source=c.source,
                url=c.url,
                page_number=c.page_number,
                score=c.score,
                rank=i,
                extra=c.extra,
            )
            for i, c in enumerate(out, start=1)
        ]
