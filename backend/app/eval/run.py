"""Evaluation harness for retrieval and system ablations.

Suites:
- ``retrieval`` — naive / hybrid / hybrid_rerank over labeled expected sources
- ``system`` — single_llm vs rag_only vs full_agent (schema, citations, tools, latency)

Artifacts are written to ``EVAL_RESULTS_DIR`` as stamped JSON plus ``latest.json``.
Run via ``python -m app.eval.run`` (see CLI flags in :func:`main`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select

from app.agents.graph import ResearchGraph
from app.agents.invoke import structured
from app.agents.prompts import DRAFT_SYSTEM
from app.agents.schemas import CitedReport
from app.config import get_settings
from app.db.models import Document
from app.db.session import SessionLocal, init_db
from app.eval.metrics import (
    citation_correctness,
    citation_precision_recall,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    tool_selection_accuracy,
)
from app.llm.models import complete
from app.retrieval.service import RetrievalService, Variant

SystemVariant = Literal["single_llm", "rag_only", "full_agent"]


async def _load_items(path: Path) -> list[dict]:
    """Load JSONL evaluation questions (one object per line)."""
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _avg(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else 0.0


def _write_results(payload: dict[str, Any]) -> Path:
    """Persist stamped + latest JSON result artifacts; return ``latest.json`` path."""
    settings = get_settings()
    out_dir = Path(settings.eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamped = out_dir / f"run_{stamp}.json"
    latest = out_dir / "latest.json"
    text = json.dumps(payload, indent=2, default=str)
    stamped.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return latest


def _format_chunks_for_prompt(chunks: list[Any]) -> str:
    """Format retrieval hits for the rag_only draft prompt."""
    blocks = []
    for i, c in enumerate(chunks[:8], start=1):
        blocks.append(f"[{i}] {c.title} (chunk {c.chunk_id})\n{c.content}")
    return "\n\n".join(blocks) if blocks else "(no retrieved chunks)"


def _citation_ids_from_report(report: dict[str, Any]) -> list[str]:
    """Collect chunk ids and URLs cited in a report for metric scoring."""
    ids: list[str] = []
    for cit in report.get("citations") or []:
        if cit.get("chunk_id"):
            ids.append(str(cit["chunk_id"]))
        if cit.get("url"):
            ids.append(str(cit["url"]))
    return ids


async def evaluate_retrieval(variant: Variant, generation: bool = False) -> dict[str, Any]:
    """Score one retrieval ablation variant against labeled expected sources."""
    settings = get_settings()
    items = await _load_items(Path(settings.eval_path))
    await init_db()
    recs: list[float] = []
    precs: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    gen_scores: list[dict] = []
    latencies: list[float] = []

    async with SessionLocal() as session:
        retriever = RetrievalService(session)
        docs = (await session.scalars(select(Document))).all()
        title_by_id = {d.id: d.title for d in docs}

        for item in items:
            expected_titles = [t.lower() for t in item.get("expected_sources") or []]
            t0 = time.perf_counter()
            chunks = await retriever.hybrid_search(item["question"], variant=variant, k=10)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            retrieved_titles = []
            for ch in chunks:
                title = (title_by_id.get(ch.document_id) or ch.title).lower()
                if title not in retrieved_titles:
                    retrieved_titles.append(title)
            recs.append(recall_at_k(retrieved_titles, expected_titles, 5))
            precs.append(precision_at_k(retrieved_titles, expected_titles, 5))
            mrrs.append(mrr(retrieved_titles, expected_titles))
            ndcgs.append(ndcg_at_k(retrieved_titles, expected_titles, 5))

            if generation:
                context = "\n\n".join(f"[{c.chunk_id}] {c.title}: {c.content}" for c in chunks[:6])
                answer = complete(
                    "Answer using only the context. Cite chunk ids in brackets.",
                    f"Question: {item['question']}\n\nContext:\n{context}",
                    role="fast",
                )
                cited = [c.chunk_id for c in chunks if c.chunk_id[:8] in answer or c.chunk_id in answer]
                judge = complete(
                    "Score 0-1 JSON with keys faithfulness, answer_relevance, context_relevance. Numbers only.",
                    f"Question: {item['question']}\nExpected: {item.get('expected_answer')}\nAnswer: {answer}\nContext: {context[:3000]}",
                    role="fast",
                )
                gen_scores.append(
                    {
                        "question": item["question"],
                        "citation_correctness": citation_correctness(cited, [c.chunk_id for c in chunks]),
                        "judge_raw": judge[:500],
                    }
                )

    return {
        "suite": "retrieval",
        "variant": variant,
        "n": len(items),
        "recall@5": _avg(recs),
        "precision@5": _avg(precs),
        "mrr": _avg(mrrs),
        "ndcg@5": _avg(ndcgs),
        "latency_p50_ms": round(percentile(latencies, 50) or 0.0, 2),
        "latency_p95_ms": round(percentile(latencies, 95) or 0.0, 2),
        "generation": gen_scores if generation else [],
    }


async def _run_single_llm(question: str) -> dict[str, Any]:
    """Ablation: bare LLM answer with no retrieval or tools."""
    t0 = time.perf_counter()
    text = complete(
        "You are a research analyst. Answer carefully. If unsure, say so. Do not invent citations.",
        question,
        role="research",
    )
    latency = (time.perf_counter() - t0) * 1000.0
    return {
        "markdown": text,
        "report": {"title": "single_llm", "citations": [], "executive_summary": text[:500]},
        "schema_valid": False,
        "route": {},
        "chunks": [],
        "web_results": [],
        "critic": {},
        "retries": 0,
        "latency_ms": latency,
        "trace": {},
    }


async def _run_rag_only(session, question: str) -> dict[str, Any]:
    """Ablation: hybrid retrieve + draft only (no web/SQL/critic loop)."""
    t0 = time.perf_counter()
    retriever = RetrievalService(session)
    chunks = await retriever.hybrid_search(question, variant="hybrid_rerank", k=8)
    prompt = (
        f"Original question:\n{question}\n\n"
        f"Retrieved chunks:\n{_format_chunks_for_prompt(chunks)}\n\n"
        "Google Scholar and paper results:\n(no web results)\n\n"
        "SQL result:\n{}\n\n"
        "Produce the cited report JSON."
    )
    report = structured(CitedReport, DRAFT_SYSTEM, prompt, role="research")
    assert isinstance(report, CitedReport)
    markdown = (
        f"# {report.title}\n\n{report.executive_summary}\n\n{report.body_markdown}\n\n"
        f"## Recommendation\n{report.recommendation}\n"
    )
    latency = (time.perf_counter() - t0) * 1000.0
    return {
        "markdown": markdown,
        "report": report.model_dump(),
        "schema_valid": True,
        "route": {"needs_retrieval": True, "needs_web": False, "needs_sql": False, "intent": "knowledge"},
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "url": c.url,
            }
            for c in chunks
        ],
        "web_results": [],
        "critic": {},
        "retries": 0,
        "latency_ms": latency,
        "trace": {},
    }


async def _run_full_agent(session, question: str) -> dict[str, Any]:
    """Ablation: full LangGraph multi-agent pipeline (production path)."""
    t0 = time.perf_counter()
    graph = ResearchGraph(session)
    final = await graph.ainvoke(question, conversation_id=None, user_id="eval-user")
    latency = (time.perf_counter() - t0) * 1000.0
    report = final.get("report") or {}
    critic = final.get("critic") or {}
    return {
        "markdown": final.get("markdown") or "",
        "report": report,
        "schema_valid": bool(report.get("title") and "citations" in report),
        "route": final.get("route") or {},
        "chunks": final.get("chunks") or [],
        "web_results": final.get("web_results") or [],
        "critic": critic,
        "retries": int(critic.get("retries") or 0) if isinstance(critic, dict) else 0,
        "latency_ms": latency,
        "trace": final.get("trace") or {},
    }


async def evaluate_system(system: SystemVariant, generation: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Score one system ablation across the eval question set."""
    settings = get_settings()
    items = await _load_items(Path(settings.eval_path))
    if limit is not None:
        items = items[:limit]
    await init_db()

    schema_ok: list[float] = []
    cit_prec: list[float] = []
    cit_rec: list[float] = []
    cit_corr: list[float] = []
    tool_acc: list[float] = []
    critic_pass: list[float] = []
    retries_list: list[float] = []
    latencies: list[float] = []
    gen_scores: list[dict] = []

    async with SessionLocal() as session:
        for item in items:
            q = item["question"]
            if system == "single_llm":
                result = await _run_single_llm(q)
            elif system == "rag_only":
                result = await _run_rag_only(session, q)
            else:
                result = await _run_full_agent(session, q)

            latencies.append(float(result["latency_ms"]))
            schema_ok.append(1.0 if result["schema_valid"] else 0.0)

            available = [c.get("chunk_id") for c in result.get("chunks") or [] if c.get("chunk_id")]
            available += [w.get("url") for w in result.get("web_results") or [] if w.get("url")]
            cited = _citation_ids_from_report(result.get("report") or {})
            # Also accept short chunk-id prefixes in citations
            expanded_avail = set(available)
            for cid in list(available):
                if cid:
                    expanded_avail.add(cid[:8])
            cited_norm = []
            for c in cited:
                cited_norm.append(c)
                if len(c) > 8:
                    cited_norm.append(c[:8])
            p, r = citation_precision_recall(cited_norm, list(expanded_avail))
            cit_prec.append(p)
            cit_rec.append(r)
            cit_corr.append(citation_correctness(cited, available) if cited else 0.0)

            acc = tool_selection_accuracy(
                {
                    "needs_sql": (result.get("route") or {}).get("needs_sql"),
                    "needs_web": (result.get("route") or {}).get("needs_web"),
                    "intent": (result.get("route") or {}).get("intent"),
                },
                {
                    "expects_sql": item.get("expects_sql"),
                    "expects_web": item.get("expects_web"),
                    "expected_intent": item.get("expected_intent"),
                },
            )
            if acc is not None:
                tool_acc.append(acc)

            critic = result.get("critic") or {}
            if system == "full_agent" and critic:
                critic_pass.append(1.0 if critic.get("pass_check") else 0.0)
                retries_list.append(float(result.get("retries") or 0))

            if generation and result.get("markdown"):
                judge = complete(
                    "Score 0-1 JSON with keys faithfulness, answer_relevance. Numbers only.",
                    f"Question: {q}\nExpected: {item.get('expected_answer')}\nAnswer: {result['markdown'][:3500]}",
                    role="fast",
                )
                gen_scores.append({"question": q, "judge_raw": judge[:500]})

    out: dict[str, Any] = {
        "suite": "system",
        "system": system,
        "n": len(items),
        "schema_validity": _avg(schema_ok),
        "citation_precision": _avg(cit_prec),
        "citation_recall": _avg(cit_rec),
        "citation_correctness": _avg(cit_corr),
        "tool_selection_accuracy": _avg(tool_acc) if tool_acc else None,
        "latency_p50_ms": round(percentile(latencies, 50) or 0.0, 2),
        "latency_p95_ms": round(percentile(latencies, 95) or 0.0, 2),
        "mean_latency_ms": round(_avg(latencies), 2),
        "generation": gen_scores if generation else [],
    }
    if system == "full_agent":
        out["critic_first_pass_rate"] = _avg(critic_pass) if critic_pass else None
        out["mean_critic_retries"] = _avg(retries_list) if retries_list else 0.0
    return out


async def run_all(
    *,
    suite: str,
    variant: Variant,
    system: SystemVariant,
    generation: bool,
    limit: int | None,
) -> dict[str, Any]:
    """Orchestrate requested suite(s), write artifacts, return the payload."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "research": settings.llm_research_model,
            "fast": settings.llm_fast_model,
            "provider": settings.llm_provider,
            "embedding": settings.embedding_model,
            "embedding_backend": settings.embedding_backend,
        },
        "eval_path": settings.eval_path,
        "results": {},
    }

    if suite in {"retrieval", "all"}:
        payload["results"]["retrieval"] = {
            variant: await evaluate_retrieval(variant, generation=generation and suite == "retrieval")
        }
        if suite == "all":
            # Full retrieval ablation matrix
            matrix = {}
            for v in ("naive", "hybrid", "hybrid_rerank"):
                matrix[v] = await evaluate_retrieval(v, generation=False)  # type: ignore[arg-type]
            payload["results"]["retrieval"] = matrix

    if suite in {"system", "all"}:
        if suite == "all":
            systems: list[SystemVariant] = ["single_llm", "rag_only", "full_agent"]
            payload["results"]["system"] = {}
            for s in systems:
                payload["results"]["system"][s] = await evaluate_system(s, generation=generation, limit=limit)
        else:
            payload["results"]["system"] = {
                system: await evaluate_system(system, generation=generation, limit=limit)
            }

    path = _write_results(payload)
    payload["artifact"] = str(path)
    return payload


def main() -> None:
    """CLI entry: ``python -m app.eval.run --suite all``."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval and/or multi-agent system ablations.")
    parser.add_argument("--suite", choices=["retrieval", "system", "all"], default="retrieval")
    parser.add_argument("--variant", choices=["naive", "hybrid", "hybrid_rerank"], default="hybrid_rerank")
    parser.add_argument("--system", choices=["single_llm", "rag_only", "full_agent"], default="full_agent")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on questions (useful for smoke tests)")
    args = parser.parse_args()
    result = asyncio.run(
        run_all(
            suite=args.suite,
            variant=args.variant,  # type: ignore[arg-type]
            system=args.system,  # type: ignore[arg-type]
            generation=args.generation,
            limit=args.limit,
        )
    )
    # Print compact summary without huge generation blobs
    printable = json.loads(json.dumps(result, default=str))
    for block in (printable.get("results") or {}).values():
        if isinstance(block, dict):
            for metrics in block.values():
                if isinstance(metrics, dict):
                    metrics.pop("generation", None)
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
