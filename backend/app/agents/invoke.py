"""LLM invocation helpers shared by agent nodes.

Provides a single entry point for structured outputs so every node gets:
- consistent model role selection (``research`` vs ``fast``)
- observability spans with token usage when available
- a JSON-parse fallback when native structured output fails (common with
  local Ollama models)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.llm.models import get_chat_model
from app.observability.tracing import extract_usage, span

logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    """Normalize LangChain message content (str or multimodal list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def structured(schema: type[BaseModel], system: str, user: str, role="research") -> BaseModel:
    """Invoke an LLM and validate the response into ``schema``.

    Prefer provider-native structured output; on failure, re-prompt for a
    single JSON object matching the schema and parse the first ``{...}`` block.
    """
    with span("llm.structured", schema=schema.__name__, role=role) as sp:
        llm = get_chat_model(role)
        try:
            bound = llm.with_structured_output(schema)
            result = bound.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            usage = extract_usage(result)
            sp.attrs.update(usage)
            sp.attrs["mode"] = "structured_output"
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            if hasattr(result, "content"):
                return schema.model_validate_json(_content_to_text(result.content))
        except Exception as exc:
            logger.warning("structured_output failed (%s); falling back to JSON parse", exc)
            sp.attrs["structured_fallback"] = True

        raw = get_chat_model(role).invoke(
            [
                SystemMessage(content=system + "\nRespond with a single JSON object matching the schema."),
                HumanMessage(content=user + f"\n\nJSON schema:\n{json.dumps(schema.model_json_schema())}"),
            ]
        )
        usage = extract_usage(raw)
        sp.attrs.update({k: v for k, v in usage.items() if v is not None})
        sp.attrs["mode"] = "json_parse"
        text = _content_to_text(raw.content if hasattr(raw, "content") else raw)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object in model output: {text[:400]}")
        return schema.model_validate_json(text[start : end + 1])
