from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(list(retrieved)[:k])
    rel = set(relevant)
    return len(top & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for item in top if item in rel) / len(top)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, item in enumerate(retrieved, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def dcg(rels: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(rels, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    gains = [1.0 if item in rel else 0.0 for item in list(retrieved)[:k]]
    ideal = sorted(gains, reverse=True)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(gains) / denom


def citation_correctness(cited_ids: Sequence[str], available_ids: Sequence[str]) -> float:
    if not cited_ids:
        return 0.0
    avail = set(available_ids)
    return sum(1 for cid in cited_ids if cid in avail) / len(cited_ids)
