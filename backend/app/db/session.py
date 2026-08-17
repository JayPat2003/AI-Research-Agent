from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, text

from app.config import get_settings
from app.db.models import Base

_settings = get_settings()

engine = create_async_engine(_settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

sync_engine = create_engine(_settings.database_url_sync, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False, class_=Session)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                """
                UPDATE chunks
                SET tsv = to_tsvector('english', coalesce(content, ''))
                WHERE tsv IS NULL
                """
            )
        )
        # Trigger to keep tsvector in sync
        await conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
                BEGIN
                  NEW.tsv := to_tsvector('english', coalesce(NEW.content, ''));
                  RETURN NEW;
                END
                $$ LANGUAGE plpgsql;
                """
            )
        )
        await conn.execute(text("DROP TRIGGER IF EXISTS tsvectorupdate ON chunks"))
        await conn.execute(
            text(
                """
                CREATE TRIGGER tsvectorupdate
                BEFORE INSERT OR UPDATE OF content ON chunks
                FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
                """
            )
        )


def init_db_sync() -> None:
    with sync_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(conn)
        conn.execute(text("DROP TRIGGER IF EXISTS tsvectorupdate ON chunks"))
        conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
                BEGIN
                  NEW.tsv := to_tsvector('english', coalesce(NEW.content, ''));
                  RETURN NEW;
                END
                $$ LANGUAGE plpgsql;
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TRIGGER tsvectorupdate
                BEFORE INSERT OR UPDATE OF content ON chunks
                FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
                """
            )
        )
