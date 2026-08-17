from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import redis

from app.config import get_settings
from app.db.session import SyncSessionLocal, init_db_sync
from app.db.seed_sql import seed_trials
from app.ingestion.pipeline import ingest_payload, seed_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")


def run() -> None:
    settings = get_settings()
    init_db_sync()
    with SyncSessionLocal() as session:
        n = seed_trials(session)
        if n:
            logger.info("Seeded %s clinical_trials rows", n)
    seeded = seed_directory(settings.seed_dir)
    if seeded:
        logger.info("Seeded %s knowledge-base documents", len(seeded))

    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    client = redis.from_url(settings.redis_url, decode_responses=True)
    logger.info("Ingestion worker listening on %s", settings.ingest_queue)
    while True:
        item = client.blpop(settings.ingest_queue, timeout=5)
        if not item:
            continue
        _, raw = item
        try:
            payload = json.loads(raw)
            ingest_payload(payload)
        except Exception:
            logger.exception("Job failed: %s", raw[:300])
            time.sleep(1)


if __name__ == "__main__":
    run()
