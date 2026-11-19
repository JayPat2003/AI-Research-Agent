"""LLM and embedding factories.

Chat models are selected by ``LLM_PROVIDER`` (Gemini primary with optional
Ollama fallback, or Ollama-only). Embeddings/reranking run locally via
``fastembed`` (or deterministic ``hash`` vectors for tests / Free Tier demos).
"""
