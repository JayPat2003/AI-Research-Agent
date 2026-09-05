"""Dense vector store: ChromaDB + local BGE embeddings."""

from __future__ import annotations

from pathlib import Path

import chromadb

from app.config import get_settings
from app.llm.embeddings import get_embedder, get_reranker
from app.retrieval.interface import RetrievedChunk


def _client():
    path = Path(get_settings().chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def collection(name: str = "research_chunks"):
    return _client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def reset(name: str = "research_chunks") -> None:
    client = _client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(rows: list[dict], name: str = "research_chunks") -> int:
    if not rows:
        return 0
    col = collection(name)
    embedder = get_embedder()
    texts = [r["content"] for r in rows]
    embeddings = embedder.embed(texts)
    ids = [r["chunk_id"] for r in rows]
    metadatas = []
    for r in rows:
        metadatas.append(
            {
                "document_id": r.get("document_id") or "",
                "title": r.get("title") or "Untitled",
                "source": r.get("source") or "",
                "url": r.get("url") or "",
                "page_number": int(r["page_number"]) if r.get("page_number") is not None else -1,
            }
        )
    col.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(rows)


def dense_search(query: str, k: int | None = None, name: str = "research_chunks") -> list[RetrievedChunk]:
    settings = get_settings()
    k = k or settings.dense_k
    col = collection(name)
    if col.count() == 0:
        return []
    embedding = get_embedder().embed_query(query)
    result = col.query(query_embeddings=[embedding], n_results=min(k, col.count()))
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    chunks: list[RetrievedChunk] = []
    for rank, (cid, text, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        meta = meta or {}
        page = meta.get("page_number")
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                document_id=str(meta.get("document_id") or ""),
                content=text or "",
                title=str(meta.get("title") or "Untitled"),
                source=str(meta.get("source") or ""),
                url=meta.get("url") or None,
                page_number=None if page in (None, -1, "-1") else int(page),
                score=1.0 - float(dist),
                rank=rank,
            )
        )
    return chunks


def rerank(query: str, chunks: list[RetrievedChunk], k: int | None = None) -> list[RetrievedChunk]:
    settings = get_settings()
    k = k or settings.rerank_k
    if not chunks:
        return []
    scores = get_reranker().score(query, [c.content for c in chunks])
    paired = sorted(zip(chunks, scores, strict=False), key=lambda item: item[1], reverse=True)
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
