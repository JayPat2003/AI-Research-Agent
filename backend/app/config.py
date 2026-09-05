"""Application settings. Secrets come from the environment / .env only."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        extra="ignore",
    )

    gemini_api_key: str = ""
    llm_provider: str = "gemini"
    llm_research_model: str = "gemini-3.5-flash"
    llm_fast_model: str = "gemini-3.5-flash-lite"

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
    chroma_path: str = "../data/chroma"
    catalog_path: str = "../data/catalog/clinical_trials.json"
    memory_dir: str = "../data/memory"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    chunk_size: int = 512
    chunk_overlap: int = 64
    dense_k: int = 20
    sparse_k: int = 20
    rerank_k: int = 8
    critic_max_retries: int = 2
    short_term_turns: int = 8
    default_user_id: str = "local-user"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
