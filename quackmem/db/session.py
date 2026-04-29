from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from quackmem.core.config import TrackerConfig
from quackmem.core.exceptions import TrackerConfigError

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def init_engine(config: TrackerConfig) -> None:
    global _engine, _session_factory
    _engine = create_async_engine(
        config.database_url,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        echo=config.echo,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise TrackerConfigError("Engine not initialized. Call init_tracker() first.")
    return _engine


@contextlib.asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    if _session_factory is None:
        raise TrackerConfigError("Session factory not initialized. Call init_tracker() first.")
    async with _session_factory() as session:
        yield session
