"""Sequential multi-agent research pipeline. Dense retrieval only."""

from __future__ import annotations

import json
from typing import Any

from app.agents.invoke import structured
from app.agents.prompts import CATALOG_SYSTEM, CRITIC_SYSTEM, DRAFT_SYSTEM, ROUTER_SYSTEM
from app.agents.schemas import CatalogFilter, CitedReport, CriticVerdict, RouteDecision
from app.config import get_settings
from app.pipeline.catalog import filter_catalog
from app.pipeline.chroma_kb import dense_search, rerank
from app.pipeline.memory import append_and_summarize, format_for_prompt, load
from app.retrieval.interface import RetrievedChunk
from app.tools.web import search_arxiv, search_semantic_scholar, search_web


def _banner(step: int, total: int, title: str, why: str) -> None:
    print()
    print("=" * 72)
    print(f"STEP {step}/{total}  {title}")
    print("-" * 72)
    print(why)
    print("=" * 72)


def _chunk_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "source": chunk.source,
        "url": chunk.url,
        "page_number": chunk.page_number,
        "score": chunk.score,
        "rank": chunk.rank,
        "content": chunk.content,
    }


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(no retrieved chunks)"
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        blocks.append(f"[{i}] {ch['title']} ({ch['chunk_id']})\n{ch['content']}")
    return "\n\n".join(blocks)


def _format_web(items: list[dict[str, Any]]) -> str:
    if not items:
        return "(no web results)"
    return "\n\n".join(
        f"W{i}. {item.get('title')} ({item.get('source')}) {item.get('url')}\n{item.get('snippet')}"
        for i, item in enumerate(items, start=1)
    )


def _to_markdown(report: CitedReport) -> str:
    md = (
        f"# {report.title}\n\n"
        f"{report.executive_summary}\n\n"
        f"{report.body_markdown}\n\n"
        f"## Recommendation\n{report.recommendation}\n"
    )
    if report.limitations:
        md += "\n## Limitations\n" + "\n".join(f"- {x}" for x in report.limitations) + "\n"
    if report.citations:
        md += "\n## Sources\n"
        for i, cit in enumerate(report.citations, start=1):
            loc = cit.url or (f"chunk:{cit.chunk_id}" if cit.chunk_id else "")
            md += f"[{i}] {cit.title} {loc}\n"
    return md


def run_query(question: str, conversation_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    total = 8

    _banner(1, total, "MEMORY", "Load prior turns + summary so follow-ups stay grounded.")
    bundle = load(conversation_id)
    memory_text = format_for_prompt(bundle)
    print(f"conversation_id: {bundle['id']}")
    print(memory_text)

    _banner(2, total, "ROUTER", "Gemini Flash-Lite decides intent and which specialists run.")
    route = structured(
        RouteDecision,
        ROUTER_SYSTEM,
        f"Memory:\n{memory_text}\n\nUser question:\n{question}",
        role="fast",
    )
    assert isinstance(route, RouteDecision)
    print(json.dumps(route.model_dump(), indent=2))

    chunks: list[dict[str, Any]] = []
    if route.needs_retrieval:
        _banner(
            3,
            total,
            "DENSE RETRIEVAL",
            "Embed the query with BGE → Chroma cosine search → cross-encoder rerank.\n"
            "Dense (not sparse/BM25): research questions are semantic, not keyword lookups.",
        )
        raw = dense_search(route.rewritten_query or question)
        ranked = rerank(route.rewritten_query or question, raw)
        chunks = [_chunk_dict(c) for c in ranked]
        print(f"chroma candidates: {len(raw)}  after rerank: {len(chunks)}")
        for c in chunks:
            print(f"  [{c['rank']}] {c['title']}  score={c['score']:.3f}  {c['chunk_id']}")
    else:
        _banner(3, total, "DENSE RETRIEVAL", "Skipped — router said this is not a knowledge-base question.")

    web_results: list[dict[str, Any]] = []
    if route.needs_web:
        _banner(4, total, "RESEARCH AGENT", "Public web + arXiv + Semantic Scholar (no paid search API).")
        q = route.rewritten_query or question
        web = search_web(q, max_results=5)
        papers = search_arxiv(q, max_results=4)
        scholar = search_semantic_scholar(q, max_results=4, api_key=settings.semantic_scholar_api_key or None)
        web_results = web + papers + scholar
        print(f"web={len(web)}  arxiv={len(papers)}  scholar={len(scholar)}")
        for item in web_results:
            print(f"  - {item.get('source')}: {item.get('title')}")
    else:
        _banner(4, total, "RESEARCH AGENT", "Skipped — private KB is enough for this question.")

    catalog_result: dict[str, Any] = {}
    if route.needs_sql:
        _banner(5, total, "DATA AGENT", "Filter the local JSON trial catalog (no database server).")
        flt = structured(
            CatalogFilter,
            CATALOG_SYSTEM,
            f"User question:\n{route.rewritten_query or question}",
            role="fast",
        )
        assert isinstance(flt, CatalogFilter)
        catalog_result = filter_catalog(flt)
        print(json.dumps({"filters": catalog_result["filters"], "n": catalog_result["n"]}, indent=2))
        for row in catalog_result.get("rows", [])[:8]:
            print(f"  - {row['trial_name']} ({row['drug_class']}, {row['status']})")
    else:
        _banner(5, total, "DATA AGENT", "Skipped — question is not a catalog count/filter.")

    def draft() -> tuple[CitedReport, str]:
        _banner(6, total, "REPORT GENERATOR", "Gemini Flash writes a cited briefing from retrieved evidence.")
        report = structured(
            CitedReport,
            DRAFT_SYSTEM,
            (
                f"Conversation memory:\n{memory_text}\n\n"
                f"Original question:\n{question}\n\n"
                f"Resolved question:\n{route.rewritten_query}\n\n"
                f"Retrieved chunks:\n{_format_chunks(chunks)}\n\n"
                f"Web and paper results:\n{_format_web(web_results)}\n\n"
                f"Catalog result:\n{json.dumps(catalog_result, default=str)[:4000]}\n"
            ),
            role="research",
        )
        assert isinstance(report, CitedReport)
        markdown = _to_markdown(report)
        print(f"title: {report.title}")
        print(f"citations: {len(report.citations)}")
        return report, markdown

    report, markdown = draft()

    retries = 0
    while True:
        _banner(7, total, "CRITIC", "Check claims, citations, hallucinations, and whether the question was answered.")
        verdict = structured(
            CriticVerdict,
            CRITIC_SYSTEM,
            (
                f"Question:\n{question}\n\nDraft:\n{markdown}\n\n"
                f"Retrieved evidence:\n{_format_chunks(chunks)}\n\n"
                f"Web evidence:\n{_format_web(web_results)}"
            ),
            role="research",
        )
        assert isinstance(verdict, CriticVerdict)
        print(json.dumps(verdict.model_dump(), indent=2))
        if verdict.pass_check or retries >= settings.critic_max_retries:
            break
        retries += 1
        extra_q = verdict.missing_queries or [route.rewritten_query or question]
        print(f"CRITIC FAIL — retrieve again ({retries}/{settings.critic_max_retries}): {extra_q}")
        seen = {c["chunk_id"]: c for c in chunks}
        for q in extra_q:
            for chunk in rerank(q, dense_search(q)):
                seen[chunk.chunk_id] = _chunk_dict(chunk)
        chunks = list(seen.values())
        report, markdown = draft()

    _banner(8, total, "FINAL REPORT", "Persist memory (summary + last turns) and return the briefing.")
    append_and_summarize(bundle, question, markdown)
    print(markdown)
    return {
        "conversation_id": bundle["id"],
        "route": route.model_dump(),
        "chunks": chunks,
        "web_results": web_results,
        "catalog": catalog_result,
        "report": report.model_dump(),
        "markdown": markdown,
        "critic": verdict.model_dump(),
        "retries": retries,
    }
