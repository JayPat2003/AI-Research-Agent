from __future__ import annotations

import json
import logging
from typing import Annotated, Any, AsyncIterator, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.invoke import structured
from app.agents.prompts import CRITIC_SYSTEM, DRAFT_SYSTEM, ROUTER_SYSTEM, SQL_SYSTEM
from app.agents.schemas import CitedReport, CriticVerdict, RouteDecision, SQLDecision
from app.config import get_settings
from app.memory.store import (
    append_turn,
    ensure_conversation,
    format_memory_for_prompt,
    load_memory_bundle,
    persist_message,
    refresh_summary,
)
from app.retrieval.interface import RetrievedChunk
from app.retrieval.service import RetrievalService
from app.tools.sql import SCHEMA_HINT, run_select
from app.tools.web import search_arxiv, search_semantic_scholar, search_web

logger = logging.getLogger(__name__)


def _extend(a: list, b: list) -> list:
    return (a or []) + (b or [])


class AgentState(TypedDict, total=False):
    query: str
    conversation_id: str
    user_id: str
    memory_text: str
    rewritten_query: str
    intent: str
    needs_retrieval: bool
    needs_web: bool
    needs_sql: bool
    extra_queries: list[str]
    chunks: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    sql_result: dict[str, Any]
    draft_markdown: str
    report: dict[str, Any]
    critic: dict[str, Any]
    retries: int
    events: Annotated[list[dict[str, Any]], _extend]


def chunk_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "source": chunk.source,
        "url": chunk.url,
        "page_number": chunk.page_number,
        "score": chunk.score,
        "content": chunk.content,
        "extra": chunk.extra,
    }


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        label = f"[{i}] {ch['title']} (chunk {ch['chunk_id'][:8]}, source={ch['source']})"
        blocks.append(f"{label}\n{ch['content']}")
    return "\n\n".join(blocks) if blocks else "(no retrieved chunks)"


def _format_web(items: list[dict[str, Any]]) -> str:
    if not items:
        return "(no web results)"
    lines = []
    for i, item in enumerate(items, start=1):
        lines.append(
            f"W{i}. {item.get('title')} ({item.get('source')}) {item.get('url')}\n{item.get('snippet')}"
        )
    return "\n\n".join(lines)


class ResearchGraph:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.retriever = RetrievalService(session)
        self.settings = get_settings()
        self.graph = self._compile()

    def _evt(self, etype: str, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"type": etype, "message": message, "data": data or {}}

    async def _node_memory(self, state: AgentState) -> dict[str, Any]:
        conv = await ensure_conversation(self.session, state.get("conversation_id"), state["user_id"])
        bundle = await load_memory_bundle(self.session, conv.id, state["user_id"])
        return {
            "conversation_id": conv.id,
            "memory_text": format_memory_for_prompt(bundle),
            "events": [self._evt("memory", "Loaded conversation memory", {"conversation_id": conv.id})],
        }

    async def _node_router(self, state: AgentState) -> dict[str, Any]:
        user = (
            f"Memory:\n{state.get('memory_text') or '(none)'}\n\n"
            f"User question:\n{state['query']}"
        )
        decision = structured(RouteDecision, ROUTER_SYSTEM, user, role="fast")
        assert isinstance(decision, RouteDecision)
        return {
            "intent": decision.intent,
            "rewritten_query": decision.rewritten_query or state["query"],
            "needs_retrieval": decision.needs_retrieval,
            "needs_web": decision.needs_web,
            "needs_sql": decision.needs_sql,
            "events": [
                self._evt(
                    "route",
                    f"Routed as {decision.intent}",
                    decision.model_dump(),
                )
            ],
        }

    async def _node_retrieve(self, state: AgentState) -> dict[str, Any]:
        if not state.get("needs_retrieval") and not state.get("extra_queries"):
            return {"chunks": state.get("chunks") or [], "events": [self._evt("retrieve", "Skipped private KB retrieval")]}
        query = state.get("rewritten_query") or state["query"]
        queries = [query, *(state.get("extra_queries") or [])]
        seen: dict[str, dict[str, Any]] = {c["chunk_id"]: c for c in (state.get("chunks") or [])}
        for q in queries:
            found = await self.retriever.hybrid_search(q, variant="hybrid_rerank")
            for chunk in found:
                seen[chunk.chunk_id] = chunk_to_dict(chunk)
        chunks = list(seen.values())
        return {
            "chunks": chunks,
            "extra_queries": [],
            "events": [self._evt("retrieve", f"Retrieved {len(chunks)} unique chunks", {"n": len(chunks)})],
        }

    async def _node_research(self, state: AgentState) -> dict[str, Any]:
        if not state.get("needs_web"):
            return {
                "web_results": state.get("web_results") or [],
                "events": [self._evt("research", "Skipped live web/paper search")],
            }
        query = state.get("rewritten_query") or state["query"]
        web = search_web(query, max_results=5)
        papers = search_arxiv(query, max_results=4)
        scholar = search_semantic_scholar(
            query, max_results=4, api_key=self.settings.semantic_scholar_api_key or None
        )
        combined = web + papers + scholar
        return {
            "web_results": combined,
            "events": [
                self._evt(
                    "research",
                    f"Found {len(web)} web, {len(papers)} arXiv, {len(scholar)} Semantic Scholar results",
                    {"n": len(combined)},
                )
            ],
        }

    async def _node_data(self, state: AgentState) -> dict[str, Any]:
        if not state.get("needs_sql"):
            return {"sql_result": {}, "events": [self._evt("data", "Skipped Text-to-SQL")]}
        query = state.get("rewritten_query") or state["query"]
        decision = structured(
            SQLDecision,
            SQL_SYSTEM,
            f"{SCHEMA_HINT}\n\nUser question:\n{query}",
            role="fast",
        )
        assert isinstance(decision, SQLDecision)
        result = await run_select(self.session, decision.sql)
        result["explanation"] = decision.explanation
        return {
            "sql_result": result,
            "events": [
                self._evt(
                    "data",
                    f"SQL {'ok' if result.get('ok') else 'failed'} — {result.get('n', 0)} rows",
                    {"sql": result.get("sql"), "ok": result.get("ok")},
                )
            ],
        }

    def _report_user_prompt(self, state: AgentState) -> str:
        sql = state.get("sql_result") or {}
        return (
            f"Conversation memory:\n{state.get('memory_text') or '(none)'}\n\n"
            f"Original question:\n{state['query']}\n\n"
            f"Resolved question:\n{state.get('rewritten_query')}\n\n"
            f"Retrieved chunks:\n{_format_chunks(state.get('chunks') or [])}\n\n"
            f"Web and paper results:\n{_format_web(state.get('web_results') or [])}\n\n"
            f"SQL result:\n{json.dumps(sql, default=str)[:4000]}\n\n"
            "Produce the cited report JSON."
        )

    async def _node_draft(self, state: AgentState) -> dict[str, Any]:
        report = structured(CitedReport, DRAFT_SYSTEM, self._report_user_prompt(state), role="research")
        assert isinstance(report, CitedReport)
        # Attach known chunk ids when the model omitted them but cited titles
        by_title = {c["title"].lower(): c for c in (state.get("chunks") or [])}
        for cit in report.citations:
            if not cit.chunk_id and cit.title:
                match = by_title.get(cit.title.lower())
                if match:
                    cit.chunk_id = match["chunk_id"]
                    cit.url = cit.url or match.get("url")
                    cit.source = cit.source or match.get("source") or ""
        markdown = (
            f"# {report.title}\n\n"
            f"{report.executive_summary}\n\n"
            f"{report.body_markdown}\n\n"
            f"## Recommendation\n{report.recommendation}\n"
        )
        if report.limitations:
            markdown += "\n## Limitations\n" + "\n".join(f"- {x}" for x in report.limitations) + "\n"
        if report.citations:
            markdown += "\n## Sources\n"
            for i, cit in enumerate(report.citations, start=1):
                loc = cit.url or (f"chunk:{cit.chunk_id}" if cit.chunk_id else "")
                markdown += f"[{i}] {cit.title} {loc}\n"
        return {
            "report": report.model_dump(),
            "draft_markdown": markdown,
            "events": [self._evt("draft", "Drafted cited report", {"title": report.title})],
        }

    async def _node_critic(self, state: AgentState) -> dict[str, Any]:
        payload = (
            f"Question:\n{state['query']}\n\n"
            f"Draft:\n{state.get('draft_markdown')}\n\n"
            f"Available chunk ids: {[c['chunk_id'] for c in (state.get('chunks') or [])]}\n\n"
            f"Retrieved evidence:\n{_format_chunks(state.get('chunks') or [])}\n\n"
            f"Web evidence:\n{_format_web(state.get('web_results') or [])}"
        )
        verdict = structured(CriticVerdict, CRITIC_SYSTEM, payload, role="research")
        assert isinstance(verdict, CriticVerdict)
        retries = int(state.get("retries") or 0)
        passed = verdict.pass_check
        if not passed:
            retries += 1
        extra = []
        if not passed:
            extra = verdict.missing_queries or [state.get("rewritten_query") or state["query"]]
        return {
            "critic": verdict.model_dump(),
            "retries": retries,
            "extra_queries": extra,
            "needs_retrieval": True if not passed else state.get("needs_retrieval", True),
            "needs_web": False if not passed else state.get("needs_web", False),
            "needs_sql": False if not passed else state.get("needs_sql", False),
            "events": [
                self._evt(
                    "critic",
                    "PASS" if passed else f"FAIL — {'; '.join(verdict.issues) or 'needs more evidence'}",
                    verdict.model_dump(),
                )
            ],
        }

    def _route_after_critic(self, state: AgentState) -> Literal["retrieve", "finalize"]:
        critic = state.get("critic") or {}
        retries = int(state.get("retries") or 0)
        if critic.get("pass_check"):
            return "finalize"
        if retries >= self.settings.critic_max_retries:
            return "finalize"
        return "retrieve"

    async def _node_finalize(self, state: AgentState) -> dict[str, Any]:
        conv_id = state["conversation_id"]
        report = state.get("report") or {}
        markdown = state.get("draft_markdown") or ""
        citations = report.get("citations") or []
        await append_turn(conv_id, "user", state["query"])
        await append_turn(conv_id, "assistant", markdown)
        await persist_message(self.session, conv_id, "user", state["query"])
        await persist_message(self.session, conv_id, "assistant", markdown, citations=citations, extra={"critic": state.get("critic")})
        bundle = await load_memory_bundle(self.session, conv_id, state["user_id"])
        await refresh_summary(self.session, conv_id, bundle, state["query"], markdown)
        await self.session.commit()
        return {
            "events": [
                self._evt(
                    "final",
                    "Report ready",
                    {
                        "conversation_id": conv_id,
                        "report": report,
                        "markdown": markdown,
                        "chunks": state.get("chunks") or [],
                        "web_results": state.get("web_results") or [],
                        "sql_result": state.get("sql_result") or {},
                        "critic": state.get("critic") or {},
                    },
                )
            ]
        }

    def _compile(self):
        graph = StateGraph(AgentState)
        graph.add_node("memory", self._node_memory)
        graph.add_node("router", self._node_router)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("research", self._node_research)
        graph.add_node("data", self._node_data)
        graph.add_node("draft", self._node_draft)
        graph.add_node("critic", self._node_critic)
        graph.add_node("finalize", self._node_finalize)
        graph.add_edge(START, "memory")
        graph.add_edge("memory", "router")
        graph.add_edge("router", "retrieve")
        graph.add_edge("retrieve", "research")
        graph.add_edge("research", "data")
        graph.add_edge("data", "draft")
        graph.add_edge("draft", "critic")
        graph.add_conditional_edges("critic", self._route_after_critic, {"retrieve": "retrieve", "finalize": "finalize"})
        graph.add_edge("finalize", END)
        return graph.compile()

    async def stream(self, query: str, conversation_id: str | None, user_id: str) -> AsyncIterator[dict[str, Any]]:
        initial: AgentState = {
            "query": query,
            "conversation_id": conversation_id or "",
            "user_id": user_id,
            "retries": 0,
            "chunks": [],
            "web_results": [],
            "extra_queries": [],
            "events": [],
        }
        async for update in self.graph.astream(initial, stream_mode="updates"):
            for _node, payload in update.items():
                for event in payload.get("events") or []:
                    yield event
