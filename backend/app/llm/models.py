"""LangChain LLM factory: Gemini primary, local Ollama fallback (no Groq)."""

from __future__ import annotations

import logging
import time
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings

logger = logging.getLogger(__name__)

Role = Literal["research", "fast"]


def _is_quota_or_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "429",
            "resource exhausted",
            "rate",
            "quota",
            "timeout",
            "unavailable",
            "503",
            "connection refused",
            "connect error",
        )
    )


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
                if not _is_quota_or_transient(exc) or attempt == self.max_retries - 1:
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


class FallbackChat:
    """Try primary (usually Gemini); on quota/transient failure use Ollama."""

    def __init__(self, primary: RetryingChat, fallback: RetryingChat) -> None:
        self.primary = primary
        self.fallback = fallback

    def invoke(self, messages, **kwargs):
        try:
            return self.primary.invoke(messages, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not _is_quota_or_transient(exc):
                raise
            logger.warning("Primary LLM failed (%s); falling back to Ollama", exc)
            return self.fallback.invoke(messages, **kwargs)

    def with_structured_output(self, schema):
        return FallbackChat(
            self.primary.with_structured_output(schema),
            self.fallback.with_structured_output(schema),
        )

    def bind_tools(self, tools):
        return FallbackChat(
            self.primary.bind_tools(tools),
            self.fallback.bind_tools(tools),
        )


def _gemini_model(role: Role) -> BaseChatModel:
    settings = get_settings()

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = (
        settings.llm_research_model
        if role == "research"
        else settings.llm_fast_model
    )

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=settings.gemini_api_key,
        temperature=0.1,
    )

def _ollama_model(role: Role) -> BaseChatModel:
    settings = get_settings()
    from langchain_ollama import ChatOllama

    model = settings.ollama_research_model if role == "research" else settings.ollama_fast_model
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )


def get_chat_model(role: Role = "research") -> RetryingChat | FallbackChat:
    settings = get_settings()
    provider = (settings.llm_provider or "gemini").strip().lower()

    if provider == "ollama":
        return RetryingChat(_ollama_model(role))

    if provider == "gemini":
        primary = RetryingChat(_gemini_model(role))
        if settings.ollama_fallback:
            return FallbackChat(primary, RetryingChat(_ollama_model(role)))
        return primary

    raise ValueError(f"Unsupported LLM_PROVIDER={settings.llm_provider!r}; use gemini or ollama")


def complete(system: str, user: str, role: Role = "research") -> str:
    llm = get_chat_model(role)
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)
