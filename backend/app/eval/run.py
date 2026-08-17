from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Document
from app.db.session import SessionLocal, init_db
from app.eval.metrics import citation_correctness, mrr, ndcg_at_k, precision_at_k, recall_at_k
from app.llm.models import complete
from app.retrieval.service import RetrievalService, Variant


async def _load_items(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


async def evaluate(variant: Variant, generation: bool = False) -> dict:
    settings = get_settings()
    path = Path(settings.eval_path)
    items = await _load_items(path)
    await init_db()
    recs: list[float] = []
    precs: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    gen_scores: list[dict] = []

    async with SessionLocal() as session:
        retriever = RetrievalService(session)
        docs = (await session.scalars(select(Document))).all()
        title_by_id = {d.id: d.title for d in docs}

        for item in items:
            expected_titles = [t.lower() for t in item.get("expected_sources") or []]
            chunks = await retriever.hybrid_search(item["question"], variant=variant, k=10)
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

    def avg(xs: list[float]) -> float:
        return round(statistics.fmean(xs), 4) if xs else 0.0

    result = {
        "variant": variant,
        "n": len(items),
        "recall@5": avg(recs),
        "precision@5": avg(precs),
        "mrr": avg(mrrs),
        "ndcg@5": avg(ndcgs),
        "generation": gen_scores,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval (and optional generation).")
    parser.add_argument("--variant", choices=["naive", "hybrid", "hybrid_rerank"], default="hybrid_rerank")
    parser.add_argument("--generation", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.variant, generation=args.generation))
    print(json.dumps({k: v for k, v in result.items() if k != "generation" or args.generation}, indent=2))


if __name__ == "__main__":
    main()
