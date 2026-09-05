"""LangChain LLM factory. Gemini API only."""

from __future__ import annotations

import logging
import time
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings

logger = logging.getLogger(__name__)

Role = Literal["research", "fast"]


class RetryingChat:
    """Thin wrapper that retries rate-limit / transient errors sequentially."""

    def __init__(self, model: BaseChatModel, max_retries: int = 4) -> None:
        self.model = model
        self.max_retries = max_retries

    def invoke(self, messages, **kwargs):
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.model.invoke(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 — classify below
                last_exc = exc
                text = str(exc).lower()
                retryable = any(
                    token in text
                    for token in ("429", "resource exhausted", "rate", "timeout", "unavailable", "503")
                )
                if not retryable or attempt == self.max_retries - 1:
                    raise
                logger.warning("LLM retry %s/%s after: %s", attempt + 1, self.max_retries, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise last_exc  # pragma: no cover

    def with_structured_output(self, schema):
        bound = self.model.with_structured_output(schema)
        return RetryingChat(bound, max_retries=self.max_retries)

    def bind_tools(self, tools):
        return RetryingChat(self.model.bind_tools(tools), max_retries=self.max_retries)


def get_chat_model(role: Role = "research") -> RetryingChat:
    settings = get_settings()
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = settings.llm_research_model if role == "research" else settings.llm_fast_model
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.gemini_api_key or None,
        temperature=0.1,
    )
    return RetryingChat(llm)


def complete(system: str, user: str, role: Role = "research") -> str:
    llm = get_chat_model(role)
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)
