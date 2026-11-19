"""Hybrid retrieval stack over PostgreSQL.

Dense search uses ``pgvector``; sparse search uses Postgres full-text search
(``TSVECTOR``). Results are fused with Reciprocal Rank Fusion and optionally
reranked with a local cross-encoder. Topical overlap filtering drops off-topic
private-KB hits so the agent can escalate to live literature when needed.
"""
