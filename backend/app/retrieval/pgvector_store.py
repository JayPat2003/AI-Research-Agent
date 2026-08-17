from __future__ import annotations

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.retrieval.interface import MetadataFilter, RetrievedChunk


def _apply_filters(stmt, filters: MetadataFilter | None):
    if not filters:
        return stmt
    if filters.domain:
        stmt = stmt.where(Document.domain.ilike(f"%{filters.domain}%"))
    if filters.source:
        stmt = stmt.where(Document.source == filters.source)
    if filters.topic:
        stmt = stmt.where(Document.topic.ilike(f"%{filters.topic}%"))
    if filters.author:
        stmt = stmt.where(Document.author.ilike(f"%{filters.author}%"))
    return stmt


def _to_chunk(row, score: float, rank: int) -> RetrievedChunk:
    chunk, document = row[0], row[1]
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=document.id,
        content=chunk.content,
        title=document.title,
        source=document.source,
        url=document.url,
        page_number=chunk.page_number,
        score=float(score),
        rank=rank,
        extra={"author": document.author, "domain": document.domain, "topic": document.topic},
    )


class PgVectorStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def dense_search(
        self,
        query_embedding: list[float],
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
        stmt: Select = (
            select(Chunk, Document, distance)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(k)
        )
        stmt = _apply_filters(stmt, filters)
        result = await self.session.execute(stmt)
        rows = result.all()
        out: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            score = 1.0 - float(row.distance)
            out.append(_to_chunk(row, score, rank))
        return out

    async def sparse_search(
        self,
        query: str,
        k: int,
        filters: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        tsq = func.plainto_tsquery("english", query)
        rank_expr = func.ts_rank_cd(Chunk.tsv, tsq).label("rank_score")
        stmt: Select = (
            select(Chunk, Document, rank_expr)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.tsv.op("@@")(tsq))
            .order_by(rank_expr.desc())
            .limit(k)
        )
        stmt = _apply_filters(stmt, filters)
        result = await self.session.execute(stmt)
        rows = result.all()
        out: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            out.append(_to_chunk(row, float(row.rank_score), rank))
        return out

    async def get_chunks(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        if not chunk_ids:
            return []
        stmt = select(Chunk, Document).join(Document, Chunk.document_id == Document.id).where(Chunk.id.in_(chunk_ids))
        result = await self.session.execute(stmt)
        return [_to_chunk(row, 0.0, i) for i, row in enumerate(result.all(), start=1)]
