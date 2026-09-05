from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.config import get_settings
from app.llm.models import complete


def _dir() -> Path:
    path = Path(get_settings().memory_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(conversation_id: str) -> Path:
    return _dir() / f"{conversation_id}.json"


def load(conversation_id: str | None) -> dict:
    cid = conversation_id or str(uuid.uuid4())
    path = _path(cid)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["id"] = cid
        return data
    return {"id": cid, "summary": "", "turns": []}


def save(bundle: dict) -> None:
    cid = bundle["id"]
    _path(cid).write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def format_for_prompt(bundle: dict) -> str:
    lines = []
    if bundle.get("summary"):
        lines.append(f"Conversation summary: {bundle['summary']}")
    turns = bundle.get("turns") or []
    if turns:
        lines.append("Recent turns:")
        for turn in turns[-get_settings().short_term_turns :]:
            lines.append(f"- {turn['role']}: {turn['content'][:500]}")
    return "\n".join(lines) if lines else "(no prior conversation)"


def append_and_summarize(bundle: dict, question: str, answer: str) -> dict:
    bundle.setdefault("turns", []).append({"role": "user", "content": question})
    bundle["turns"].append({"role": "assistant", "content": answer[:4000]})
    bundle["turns"] = bundle["turns"][-get_settings().short_term_turns :]
    try:
        bundle["summary"] = complete(
            "Compress a research conversation. Be factual.",
            f"Existing summary:\n{bundle.get('summary') or '(none)'}\n\n"
            f"Latest Q:\n{question}\n\nLatest A (truncated):\n{answer[:2000]}\n\n"
            "Write an updated 4-8 sentence summary.",
            role="fast",
        ).strip()
    except Exception:
        bundle["summary"] = ((bundle.get("summary") or "") + f"\n{question[:160]}").strip()[-2000:]
    save(bundle)
    return bundle
