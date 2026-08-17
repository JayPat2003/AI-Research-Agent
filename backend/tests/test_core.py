from __future__ import annotations

from app.eval.metrics import citation_correctness, mrr, ndcg_at_k, precision_at_k, recall_at_k
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.interface import RetrievedChunk
from app.tools.sql_guard import UnsafeSQLError, validate_select


def _chunk(cid: str, rank: int, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="d",
        content=cid,
        title=cid,
        source="test",
        url=None,
        page_number=None,
        score=score,
        rank=rank,
    )


def test_recall_precision_mrr_ndcg():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["c", "z"]
    assert recall_at_k(retrieved, relevant, 5) == 0.5
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert mrr(retrieved, relevant) == 1 / 3
    assert ndcg_at_k(retrieved, ["a"], 5) == 1.0
    assert citation_correctness(["a", "x"], ["a", "b"]) == 0.5


def test_reciprocal_rank_fusion_prefers_consensus():
    left = [_chunk("a", 1), _chunk("b", 2), _chunk("c", 3)]
    right = [_chunk("c", 1), _chunk("a", 2), _chunk("d", 3)]
    fused = reciprocal_rank_fusion([left, right])
    assert fused[0].chunk_id in {"a", "c"}
    ids = [c.chunk_id for c in fused]
    assert ids.index("a") < ids.index("b")


def test_sql_guard_allows_select_and_adds_limit():
    sql = validate_select("SELECT trial_name, drug_class FROM clinical_trials WHERE status = 'completed'")
    assert "LIMIT" in sql.upper()
    assert "clinical_trials" in sql.lower()


def test_sql_guard_rejects_mutation_and_unknown_table():
    try:
        validate_select("DELETE FROM clinical_trials")
        raise AssertionError("should have failed")
    except UnsafeSQLError:
        pass
    try:
        validate_select("SELECT * FROM users")
        raise AssertionError("should have failed")
    except UnsafeSQLError:
        pass
    try:
        validate_select("SELECT 1; SELECT * FROM clinical_trials")
        raise AssertionError("should have failed")
    except UnsafeSQLError:
        pass


def test_chunking_produces_overlap_units():
    from llama_index.core import Document
    from app.ingestion.loaders import chunk_documents

    text = " ".join(f"Sentence number {i} about GLP-1 and SGLT2 outcomes trials." for i in range(80))
    chunks = chunk_documents([Document(text=text, metadata={"title": "t", "source": "md"})])
    assert len(chunks) >= 2
    assert chunks[0]["title"] == "t"
