from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, Document, IngestJob
from app.db.session import SyncSessionLocal, init_db_sync
from app.ingestion.loaders import chunk_documents, load_arxiv, load_file, load_markdown_text, load_url
from app.llm.embeddings import get_embedder

logger = logging.getLogger(__name__)


def _as_documents(payload: dict[str, Any]):
    source_type = payload["source_type"]
    source = payload["source"]
    if source_type == "file":
        return load_file(source)
    if source_type == "url":
        return load_url(source)
    if source_type == "arxiv":
        return load_arxiv(source)
    if source_type == "text":
        return load_markdown_text(source, title=payload.get("title") or "Pasted note")
    raise ValueError(f"Unknown source_type {source_type}")


def ingest_payload(payload: dict[str, Any], session: Session | None = None) -> str:
    """Synchronous ingestion used by the worker and seed script."""
    close = False
    if session is None:
        init_db_sync()
        session = SyncSessionLocal()
        close = True
    job_id = payload.get("job_id")
    job: IngestJob | None = None
    if job_id:
        job = session.get(IngestJob, job_id)
        if job:
            job.status = "processing"
            session.commit()
    try:
        docs = _as_documents(payload)
        chunks = chunk_documents(docs)
        if not chunks:
            raise ValueError("No chunks produced")
        head = chunks[0]
        document = Document(
            title=payload.get("title") or head["title"],
            author=payload.get("author") or head["author"],
            source=head["source"],
            publication_date=head["publication_date"],
            domain=payload.get("domain") or head["domain"],
            topic=payload.get("topic") or head["topic"],
            url=payload.get("url") or head["url"],
            extra={"source_type": payload["source_type"]},
        )
        session.add(document)
        session.flush()
        embedder = get_embedder()
        embeddings = embedder.embed([c["content"] for c in chunks])
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            page = chunk["page_number"]
            if isinstance(page, str) and page.isdigit():
                page = int(page)
            elif not isinstance(page, int):
                page = None
            session.add(
                Chunk(
                    document_id=document.id,
                    chunk_index=chunk["chunk_index"],
                    page_number=page,
                    content=chunk["content"],
                    embedding=embedding,
                    extra={
                        "title": chunk["title"],
                        "source": chunk["source"],
                        "url": chunk["url"],
                    },
                )
            )
        if job:
            job.status = "done"
            job.document_id = document.id
        session.commit()
        logger.info("Ingested document %s (%s chunks)", document.id, len(chunks))
        return document.id
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        if job_id:
            job = session.get(IngestJob, job_id)
            if job:
                job.status = "failed"
                job.error = str(exc)
                session.commit()
        logger.exception("Ingest failed: %s", exc)
        raise
    finally:
        if close:
            session.close()


async def enqueue_job(redis_client, payload: dict[str, Any]) -> None:
    await redis_client.rpush(get_settings().ingest_queue, json.dumps(payload))


def seed_directory(seed_dir: str | Path | None = None) -> list[str]:
    settings = get_settings()
    root = Path(seed_dir or settings.seed_dir)
    if not root.exists():
        logger.warning("Seed directory missing: %s", root)
        return []
    ids: list[str] = []
    with SyncSessionLocal() as session:
        for path in sorted(root.glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf", ".docx", ".html"}:
                continue
            found = session.scalars(select(Document).where(Document.title == path.stem)).first()
            if found:
                continue
            doc_id = ingest_payload({"source_type": "file", "source": str(path), "title": path.stem}, session)
            ids.append(doc_id)
    return ids
