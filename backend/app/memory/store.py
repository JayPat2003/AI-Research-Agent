from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Conversation, Message, UserPreference
from app.llm.models import complete

_redis: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis


def _turns_key(conversation_id: str) -> str:
    return f"conv:{conversation_id}:turns"


async def ensure_conversation(session: AsyncSession, conversation_id: str | None, user_id: str) -> Conversation:
    if conversation_id:
        conv = await session.get(Conversation, conversation_id)
        if conv:
            return conv
    conv = Conversation(user_id=user_id)
    session.add(conv)
    await session.flush()
    pref = await session.get(UserPreference, user_id)
    if pref is None:
        session.add(UserPreference(user_id=user_id, domain=None, citation_style="inline"))
        await session.flush()
    return conv


async def append_turn(conversation_id: str, role: str, content: str) -> None:
    settings = get_settings()
    client = await get_redis()
    item = json.dumps({"role": role, "content": content[:4000]})
    key = _turns_key(conversation_id)
    await client.rpush(key, item)
    await client.ltrim(key, -settings.short_term_turns, -1)
    await client.expire(key, 60 * 60 * 24 * 7)


async def recent_turns(conversation_id: str) -> list[dict[str, str]]:
    client = await get_redis()
    raw = await client.lrange(_turns_key(conversation_id), 0, -1)
    return [json.loads(item) for item in raw]


async def persist_message(
    session: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    citations: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations or [],
        extra=extra or {},
    )
    session.add(msg)
    await session.flush()
    return msg


async def load_memory_bundle(session: AsyncSession, conversation_id: str, user_id: str) -> dict[str, Any]:
    conv = await session.get(Conversation, conversation_id)
    pref = await session.get(UserPreference, user_id)
    turns = await recent_turns(conversation_id)
    return {
        "summary": conv.summary if conv else "",
        "turns": turns,
        "preferences": {
            "domain": pref.domain if pref else None,
            "citation_style": pref.citation_style if pref else "inline",
        },
    }


def format_memory_for_prompt(bundle: dict[str, Any]) -> str:
    lines = []
    prefs = bundle.get("preferences") or {}
    if prefs.get("domain"):
        lines.append(f"User domain preference: {prefs['domain']}")
    if bundle.get("summary"):
        lines.append(f"Conversation summary: {bundle['summary']}")
    turns = bundle.get("turns") or []
    if turns:
        lines.append("Recent turns:")
        for turn in turns:
            lines.append(f"- {turn['role']}: {turn['content'][:500]}")
    return "\n".join(lines)


async def refresh_summary(session: AsyncSession, conversation_id: str, bundle: dict[str, Any], latest_q: str, latest_a: str) -> None:
    conv = await session.get(Conversation, conversation_id)
    if conv is None:
        return
    prompt = (
        f"Existing summary:\n{conv.summary or '(none)'}\n\n"
        f"Latest question:\n{latest_q}\n\n"
        f"Latest answer (truncated):\n{latest_a[:2500]}\n\n"
        "Write an updated 4-8 sentence conversation summary that preserves the research topic, "
        "options being compared, and user constraints. Do not add facts that were not discussed."
    )
    try:
        summary = complete(
            "You compress research conversations. Be factual and concise.",
            prompt,
            role="fast",
        )
        conv.summary = summary.strip()
    except Exception:
        # Memory must not fail the user-facing report
        snippet = f"{latest_q[:200]} → {latest_a[:200]}"
        conv.summary = ((conv.summary or "") + "\n" + snippet).strip()[-4000:]
