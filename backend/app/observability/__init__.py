"""Local-first tracing with optional Langfuse export.

Every research run can emit JSONL traces under ``data/traces/`` without SaaS
dependencies. When ``OBSERVABILITY_BACKEND`` is ``langfuse`` or ``both`` and
API keys are present, the same spans are exported to Langfuse.
"""

from app.observability.tracing import TraceContext, current_trace, get_or_create_trace, span

__all__ = ["TraceContext", "current_trace", "get_or_create_trace", "span"]
