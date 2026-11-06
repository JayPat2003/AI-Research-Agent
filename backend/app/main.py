from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db.seed_sql import seed_trials_async
from app.db.session import SessionLocal, init_db
from app.ingestion.pipeline import seed_directory

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()
    async with SessionLocal() as session:
        n = await seed_trials_async(session)
        if n:
            logger.info("Seeded clinical_trials (%s rows)", n)
    try:
        seeded = await asyncio.to_thread(seed_directory)
        if seeded:
            logger.info("Seeded knowledge base documents: %s", len(seeded))
    except Exception:
        logger.exception("Knowledge-base seed skipped (worker may ingest instead)")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    origins = settings.cors_origin_list
    # Browsers reject credentials with wildcard origins; same-origin Caddy deploys often use "*".
    allow_credentials = "*" not in origins
    app = FastAPI(title="AI Research & Decision Intelligence Platform", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
