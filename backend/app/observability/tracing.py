"""Local-first observability: JSONL traces + optional Langfuse export.

Design goals:
- Always record per-query spans under ``TRACE_DIR`` (no cloud required).
- Optionally mirror to Langfuse when ``OBSERVABILITY_BACKEND`` is ``langfuse``
  or ``both`` and API keys are set.
- Attach a compact summary to the SSE ``final`` event for UI / eval latency.

Use :func:`span` as a context manager around graph nodes, LLM calls, and tools.
"""

from __future__ import annotations

import contextvars
import json
import logging
import statistics
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "trace_context", default=None
)


def _now_ms() -> float:
    """Monotonic clock in milliseconds (for span latency, not wall time)."""
    return time.perf_counter() * 1000.0


def _trace_dir() -> Path:
    """Ensure and return the local trace output directory."""
    from app.config import get_settings

    settings = get_settings()
    root = Path(settings.trace_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class SpanRecord:
    """One timed operation within a query run."""

    name: str
    start_ms: float
    end_ms: float | None = None
    latency_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def close(self, *, status: str = "ok", error: str | None = None, **attrs: Any) -> None:
        """Finalize latency and optional error/attr updates."""
        self.end_ms = _now_ms()
        self.latency_ms = round(self.end_ms - self.start_ms, 2)
        self.status = status
        self.error = error
        if attrs:
            self.attrs.update(attrs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL / SSE payloads."""
        return {
            "name": self.name,
            "start_ms": round(self.start_ms, 2),
            "end_ms": round(self.end_ms, 2) if self.end_ms is not None else None,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
            "attrs": self.attrs,
        }


class TraceContext:
    """Per-query trace: collects spans, aggregates latency, flushes to disk/cloud."""

    def __init__(self, run_id: str | None = None, query: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.query = query or ""
        self.started_at = time.time()
        self._t0 = _now_ms()
        self.spans: list[SpanRecord] = []
        self._langfuse_trace = None
        self._langfuse = None
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        """Best-effort Langfuse client init; never raises into the request path."""
        from app.config import get_settings

        settings = get_settings()
        backend = (settings.observability_backend or "local").strip().lower()
        if backend not in {"langfuse", "both"}:
            return
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            return
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host or "https://cloud.langfuse.com",
            )
            self._langfuse_trace = self._langfuse.trace(
                id=self.run_id,
                name="research_query",
                input={"query": self.query[:2000]},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse init failed: %s", exc)
            self._langfuse = None
            self._langfuse_trace = None

    def add_span(self, span: SpanRecord) -> None:
        """Append a closed span and optionally export it to Langfuse."""
        self.spans.append(span)
        if self._langfuse_trace is not None:
            try:
                self._langfuse_trace.span(
                    name=span.name,
                    start_time=None,
                    end_time=None,
                    metadata={
                        **span.attrs,
                        "latency_ms": span.latency_ms,
                        "status": span.status,
                        "error": span.error,
                    },
                    level="ERROR" if span.status != "ok" else "DEFAULT",
                    status_message=span.error,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Langfuse span export failed: %s", exc)

    def summary(self) -> dict[str, Any]:
        """Compact run metrics for SSE final events and eval artifacts."""
        latencies = [s.latency_ms for s in self.spans if s.latency_ms is not None]
        total = round(_now_ms() - self._t0, 2)
        by_name: dict[str, list[float]] = {}
        for s in self.spans:
            if s.latency_ms is None:
                continue
            by_name.setdefault(s.name, []).append(s.latency_ms)
        node_totals = {k: round(sum(v), 2) for k, v in by_name.items()}
        tokens_in = sum(int(s.attrs.get("input_tokens") or 0) for s in self.spans)
        tokens_out = sum(int(s.attrs.get("output_tokens") or 0) for s in self.spans)
        return {
            "run_id": self.run_id,
            "total_latency_ms": total,
            "span_count": len(self.spans),
            "p50_span_ms": round(statistics.median(latencies), 2) if latencies else None,
            "p95_span_ms": (
                round(statistics.quantiles(latencies, n=20)[18], 2)
                if len(latencies) >= 20
                else (max(latencies) if latencies else None)
            ),
            "node_latency_ms": node_totals,
            "input_tokens": tokens_in or None,
            "output_tokens": tokens_out or None,
            "errors": [s.to_dict() for s in self.spans if s.status != "ok"],
        }

    def flush(self) -> Path | None:
        """Write JSONL under ``TRACE_DIR`` and/or flush Langfuse; return local path."""
        from app.config import get_settings

        settings = get_settings()
        backend = (settings.observability_backend or "local").strip().lower()
        path: Path | None = None
        if backend in {"local", "both", ""}:
            path = _trace_dir() / f"{self.run_id}.jsonl"
            payload = {
                "run_id": self.run_id,
                "query": self.query,
                "started_at": self.started_at,
                "summary": self.summary(),
                "spans": [s.to_dict() for s in self.spans],
            }
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
                for span in self.spans:
                    fh.write(json.dumps({"type": "span", **span.to_dict()}, default=str) + "\n")
        if self._langfuse is not None:
            try:
                if self._langfuse_trace is not None:
                    self._langfuse_trace.update(output=self.summary())
                self._langfuse.flush()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse flush failed: %s", exc)
        return path


def current_trace() -> TraceContext | None:
    """Return the active context-var trace, if any."""
    return _current.get()


def get_or_create_trace(query: str | None = None, run_id: str | None = None) -> TraceContext:
    """Reuse the active trace or create and bind a new one for this request."""
    existing = _current.get()
    if existing is not None:
        return existing
    ctx = TraceContext(run_id=run_id, query=query)
    _current.set(ctx)
    return ctx


def set_current_trace(ctx: TraceContext | None) -> None:
    """Bind or clear the request-scoped trace context variable."""
    _current.set(ctx)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[SpanRecord]:
    """Record a timed span on the current trace (no-op if no active trace)."""
    record = SpanRecord(name=name, start_ms=_now_ms(), attrs=dict(attrs))
    try:
        yield record
        if record.end_ms is None:
            record.close(status="ok")
    except Exception as exc:
        record.close(status="error", error=str(exc))
        raise
    finally:
        ctx = _current.get()
        if ctx is not None:
            if record.end_ms is None:
                record.close()
            ctx.add_span(record)


def extract_usage(response: Any) -> dict[str, Any]:
    """Best-effort token usage from LangChain / provider response metadata."""
    out: dict[str, Any] = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("usage_metadata") or meta.get("token_usage") or meta.get("usage") or {}
    if not usage and hasattr(response, "usage_metadata"):
        usage = getattr(response, "usage_metadata") or {}
    if isinstance(usage, dict):
        out["input_tokens"] = usage.get("input_tokens") or usage.get("prompt_tokens")
        out["output_tokens"] = usage.get("output_tokens") or usage.get("completion_tokens")
        out["total_tokens"] = usage.get("total_tokens")
    return out
