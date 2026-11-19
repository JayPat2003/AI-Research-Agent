"""Centralized runtime configuration.

All secrets and tunables are loaded from environment variables (and optional
``.env``). Call :func:`get_settings` rather than constructing :class:`Settings`
directly so values are cached for the process lifetime.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Application settings. Secrets come from the environment / ``.env`` only."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
    )

    gemini_api_key: str = ""
    # gemini | ollama — gemini uses local Ollama as automatic quota fallback
    llm_provider: str = "gemini"
    llm_research_model: str = "gemini-3.5-flash"
    llm_fast_model: str = "gemini-3.5-flash-lite"

    ollama_base_url: str = "http://localhost:11434"
    # Research: stronger local model for draft/critic. Fast: router/SQL/summary.
    ollama_research_model: str = "qwen2.5:7b"
    ollama_fast_model: str = "llama3.2:latest"
    ollama_fallback: bool = True

    embedding_backend: str = "bge"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_dim: int = 384

    database_url: str = "postgresql+asyncpg://research:research@localhost:5432/research"
    database_url_sync: str = "postgresql+psycopg://research:research@localhost:5432/research"
    redis_url: str = "redis://localhost:6379/0"

    semantic_scholar_api_key: str = ""
    tavily_api_key: str = ""

    hf_home: str = "./models"
    ingest_queue: str = "ingest_jobs"
    upload_dir: str = "../data/uploads"
    seed_dir: str = "../data/seed"
    eval_path: str = "../data/eval/questions.jsonl"
    eval_results_dir: str = "../eval_results"
    chroma_path: str = "../data/chroma"
    catalog_path: str = "../data/catalog/clinical_trials.json"
    memory_dir: str = "../data/memory"
    trace_dir: str = "../data/traces"

    # Observability: local JSONL always; Langfuse when keys + backend allow it.
    observability_backend: str = "local"  # local | langfuse | both
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    chunk_size: int = 512
    chunk_overlap: int = 64
    dense_k: int = 20
    sparse_k: int = 20
    rerank_k: int = 8
    # Drop KB chunks below this rerank/overlap score so off-topic seed docs
    # (e.g. GLP-1 corpus vs a heart-attack question) do not pollute the draft.
    retrieval_min_score: float = 0.0
    critic_max_retries: int = 2
    short_term_turns: int = 8
    default_user_id: str = "local-user"

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated ``CORS_ORIGINS`` into a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
