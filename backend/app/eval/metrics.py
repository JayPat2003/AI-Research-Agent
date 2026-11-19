"""Retrieval and system-ablation metric helpers.

Used by :mod:`app.eval.run` for recall/precision/MRR/nDCG, citation quality,
latency percentiles, and router tool-selection accuracy against labeled
expectations in ``data/eval/questions.jsonl``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant items appearing in the top-k retrieved list."""
    if not relevant:
        return 0.0
    top = set(list(retrieved)[:k])
    rel = set(relevant)
    return len(top & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for item in top if item in rel) / len(top)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Mean Reciprocal Rank of the first relevant hit (0 if none)."""
    rel = set(relevant)
    for i, item in enumerate(retrieved, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def dcg(rels: Sequence[float]) -> float:
    """Discounted cumulative gain for a graded relevance list."""
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(rels, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalized DCG@k with binary relevance."""
    rel = set(relevant)
    gains = [1.0 if item in rel else 0.0 for item in list(retrieved)[:k]]
    ideal = sorted(gains, reverse=True)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(gains) / denom


def citation_correctness(cited_ids: Sequence[str], available_ids: Sequence[str]) -> float:
    """Fraction of cited IDs that appear in the available evidence set."""
    if not cited_ids:
        return 0.0
    avail = set(available_ids)
    return sum(1 for cid in cited_ids if cid in avail) / len(cited_ids)


def citation_precision_recall(
    cited: Sequence[str],
    available: Sequence[str],
) -> tuple[float, float]:
    """Precision/recall of cited evidence IDs against available chunk/URL evidence."""
    cited_set = {c for c in cited if c}
    avail_set = {a for a in available if a}
    if not cited_set:
        return 0.0, 0.0
    if not avail_set:
        return 0.0, 0.0
    hits = cited_set & avail_set
    precision = len(hits) / len(cited_set)
    recall = len(hits) / len(avail_set) if avail_set else 0.0
    return precision, recall


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (``p`` in 0–100)."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def tool_selection_accuracy(
    predicted: dict[str, bool | str | None],
    expected: dict[str, bool | str | None],
) -> float | None:
    """Fraction of tool/intent expectations that match. None if no expectations."""
    checks: list[bool] = []
    if "expects_sql" in expected and expected["expects_sql"] is not None:
        checks.append(bool(predicted.get("needs_sql")) == bool(expected["expects_sql"]))
    if "expects_web" in expected and expected["expects_web"] is not None:
        checks.append(bool(predicted.get("needs_web")) == bool(expected["expects_web"]))
    if expected.get("expected_intent"):
        checks.append(str(predicted.get("intent") or "") == str(expected["expected_intent"]))
    if not checks:
        return None
    return sum(1 for c in checks if c) / len(checks)
