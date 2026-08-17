from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import ResearchGraph
from app.api.schemas import (
    IngestArxivRequest,
    IngestTextRequest,
    IngestUrlRequest,
    PreferenceUpdate,
    QueryRequest,
)
from app.config import get_settings
from app.db.models import Conversation, Document, IngestJob, Message, UserPreference
from app.db.session import get_session
from app.ingestion.pipeline import enqueue_job
from app.memory.store import get_redis

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query")
async def query(body: QueryRequest, session: AsyncSession = Depends(get_session)):
    settings = get_settings()
    user_id = body.user_id or settings.default_user_id
    graph = ResearchGraph(session)

    async def events():
        try:
            async for event in graph.stream(body.query, body.conversation_id, user_id):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Query failed")
            err = {"type": "error", "message": str(exc), "data": {}}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _queue(session: AsyncSession, source_type: str, source: str, extra: dict | None = None) -> IngestJob:
    job = IngestJob(id=str(uuid.uuid4()), status="queued", source_type=source_type, source=source, extra=extra or {})
    session.add(job)
    await session.commit()
    client = await get_redis()
    await enqueue_job(
        client,
        {
            "job_id": job.id,
            "source_type": source_type,
            "source": source,
            **(extra or {}),
        },
    )
    return job


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), session: AsyncSession = Depends(get_session)):
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4()}_{file.filename}"
    dest.write_bytes(await file.read())
    job = await _queue(session, "file", str(dest), {"title": Path(file.filename or "upload").stem})
    return {"job_id": job.id, "status": job.status}


@router.post("/ingest/url")
async def ingest_url(body: IngestUrlRequest, session: AsyncSession = Depends(get_session)):
    job = await _queue(session, "url", body.url, {"domain": body.domain, "topic": body.topic, "url": body.url})
    return {"job_id": job.id, "status": job.status}


@router.post("/ingest/arxiv")
async def ingest_arxiv(body: IngestArxivRequest, session: AsyncSession = Depends(get_session)):
    job = await _queue(session, "arxiv", body.arxiv_id, {"domain": body.domain})
    return {"job_id": job.id, "status": job.status}


@router.post("/ingest/text")
async def ingest_text(body: IngestTextRequest, session: AsyncSession = Depends(get_session)):
    job = await _queue(session, "text", body.text, {"title": body.title, "domain": body.domain})
    return {"job_id": job.id, "status": job.status}


@router.get("/ingest/{job_id}")
async def ingest_status(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await session.get(IngestJob, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "error": job.error,
        "document_id": job.document_id,
    }


@router.get("/documents")
async def list_documents(session: AsyncSession = Depends(get_session)):
    rows = (await session.scalars(select(Document).order_by(Document.created_at.desc()))).all()
    return [
        {
            "id": d.id,
            "title": d.title,
            "author": d.author,
            "source": d.source,
            "domain": d.domain,
            "topic": d.topic,
            "url": d.url,
            "publication_date": d.publication_date,
        }
        for d in rows
    ]


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, session: AsyncSession = Depends(get_session)):
    conv = await session.get(Conversation, conversation_id)
    if not conv:
        raise HTTPException(404, "conversation not found")
    msgs = (
        await session.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at))
    ).all()
    return {
        "id": conv.id,
        "summary": conv.summary,
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "citations": m.citations} for m in msgs
        ],
    }


@router.put("/preferences/{user_id}")
async def update_preferences(user_id: str, body: PreferenceUpdate, session: AsyncSession = Depends(get_session)):
    pref = await session.get(UserPreference, user_id)
    if pref is None:
        pref = UserPreference(user_id=user_id)
        session.add(pref)
    if body.domain is not None:
        pref.domain = body.domain
    if body.citation_style is not None:
        pref.citation_style = body.citation_style
    if body.extra:
        pref.extra = {**(pref.extra or {}), **body.extra}
    await session.commit()
    return {"user_id": pref.user_id, "domain": pref.domain, "citation_style": pref.citation_style}
