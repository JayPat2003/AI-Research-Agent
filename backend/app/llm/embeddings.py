from __future__ import annotations

import hashlib
import logging
import threading
from typing import Sequence

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_embedder = None
_reranker = None


def _hash_vector(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec) or 1.0
    return (vec / norm).tolist()


class Embedder:
    """Local BGE embeddings via fastembed, or a hash fallback for tests."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.backend = self.settings.embedding_backend
        self.dim = self.settings.embedding_dim
        self._model = None

    def _load(self):
        if self.backend == "hash":
            return None
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("Loading embedding model %s", self.settings.embedding_model)
            self._model = TextEmbedding(model_name=self.settings.embedding_model)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.backend == "hash":
            return [_hash_vector(t, self.dim) for t in texts]
        model = self._load()
        vectors = list(model.embed(list(texts)))
        return [np.asarray(v, dtype=np.float32).tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # BGE retrieval: prefix queries
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        if self.backend == "hash":
            return _hash_vector(prefixed, self.dim)
        return self.embed([prefixed])[0]


class Reranker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    def _load(self):
        if self.settings.embedding_backend == "hash":
            return None
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            model_name = self.settings.reranker_model
            logger.info("Loading reranker %s", model_name)
            try:
                self._model = TextCrossEncoder(model_name=model_name)
            except Exception:
                fallback = "Xenova/ms-marco-MiniLM-L-6-v2"
                logger.warning("Reranker %s unavailable, falling back to %s", model_name, fallback)
                self._model = TextCrossEncoder(model_name=fallback)
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        if self.settings.embedding_backend == "hash":
            q_tokens = set(query.lower().split())
            scores = []
            for doc in documents:
                d_tokens = set(doc.lower().split())
                overlap = len(q_tokens & d_tokens)
                scores.append(float(overlap) / (len(q_tokens) + 1))
            return scores
        model = self._load()
        raw = list(model.rerank(query, list(documents)))
        # fastembed may return scores or ranked pairs — normalize to scores aligned with input
        if raw and isinstance(raw[0], tuple):
            scores = [0.0] * len(documents)
            for item in raw:
                idx = int(item[0])
                scores[idx] = float(item[1])
            return scores
        return [float(s) for s in raw]


def get_embedder() -> Embedder:
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = Embedder()
        return _embedder


def get_reranker() -> Reranker:
    global _reranker
    with _lock:
        if _reranker is None:
            _reranker = Reranker()
        return _reranker
