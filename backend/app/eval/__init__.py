"""Evaluation harness for retrieval ablation and system ablation.

Compare ``naive`` / ``hybrid`` / ``hybrid_rerank`` retrieval variants and
``single_llm`` / ``rag_only`` / ``full_agent`` system variants. Metrics and CLI
entrypoints live in :mod:`app.eval.metrics` and :mod:`app.eval.run`.
"""
