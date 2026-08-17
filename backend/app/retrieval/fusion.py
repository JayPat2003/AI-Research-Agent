from __future__ import annotations

from collections import defaultdict

from app.retrieval.interface import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Combine ranked lists with classic RRF: sum 1/(k + rank)."""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for chunk in ranking:
            scores[chunk.chunk_id] += 1.0 / (k + chunk.rank)
            current = best.get(chunk.chunk_id)
            if current is None or chunk.score > current.score:
                best[chunk.chunk_id] = chunk
    fused = []
    for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        chunk = best[cid]
        fused.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                title=chunk.title,
                source=chunk.source,
                url=chunk.url,
                page_number=chunk.page_number,
                score=score,
                extra=chunk.extra,
            )
        )
    for i, chunk in enumerate(fused, start=1):
        chunk.rank = i
    return fused
