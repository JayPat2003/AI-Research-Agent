from __future__ import annotations

from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.llm.embeddings import get_embedder, get_reranker
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.interface import MetadataFilter, RetrievedChunk
from app.retrieval.pgvector_store import PgVectorStore

Variant = Literal["naive", "hybrid", "hybrid_rerank"]


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
            return dense[:final_k]

        sparse = await self.keyword_search(query, sparse_k, filters)
        fused = reciprocal_rank_fusion([dense, sparse])
        if variant == "hybrid":
            return fused[:final_k]

        return self.rerank(query, fused, final_k)

    def rerank(self, query: str, chunks: list[RetrievedChunk], k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.rerank_k
        if not chunks:
            return []
        scores = get_reranker().score(query, [c.content for c in chunks])
        paired = list(zip(chunks, scores, strict=False))
        paired.sort(key=lambda item: item[1], reverse=True)
        out = []
        for rank, (chunk, score) in enumerate(paired[:k], start=1):
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
                    rank=rank,
                    extra=chunk.extra,
                )
            )
        return out
