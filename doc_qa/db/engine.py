"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url(config_url: str | None = None) -> str | None:
    """Resolve database URL: env var > config > None."""
    url = os.environ.get("DOC_QA_DATABASE_URL") or config_url
    if not url:
        return None
    # Ensure asyncpg driver prefix
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def init_engine(url: str) -> None:
    """Create the async engine and session factory."""
    global _engine, _session_factory

    _engine = create_async_engine(
        url,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    logger.info("Database engine initialized")


async def close_engine() -> None:
    """Dispose the engine and release all connections."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database engine closed")


def get_session() -> AsyncSession:
    """Return a new async session. Must be used as async context manager."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_engine() first")
    return _session_factory()
