"""Multi-agent orchestration layer (LangGraph).

Public entry point is :class:`ResearchGraph`, which compiles the research
workflow (memory → router → retrieve → research → data → draft → critic →
finalize) and streams node events for the SSE API. Structured I/O contracts
live in :mod:`app.agents.schemas`; system prompts in :mod:`app.agents.prompts`.
"""

from app.agents.graph import ResearchGraph
from app.agents.schemas import CitedReport, CriticVerdict, RouteDecision

__all__ = ["ResearchGraph", "CitedReport", "CriticVerdict", "RouteDecision"]
