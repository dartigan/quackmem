from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import sqlalchemy as sa
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
        # Recycle stale connections before handing them to the application.
        # Prevents errors after network hiccups or Postgres restarts.
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def verify_engine() -> None:
    """Assert the database is reachable by issuing a lightweight query.

    Call this once at startup after init_engine() (or init_tracker()) to get
    a clear TrackerConfigError if the database is unreachable, rather than
    discovering the problem silently at the first tracked call.

    Raises:
        TrackerConfigError: If the engine has not been initialised or the
            database cannot be reached.
    """
    engine = get_engine()  # raises TrackerConfigError if not initialised
    try:
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
    except Exception as exc:
        raise TrackerConfigError(
            f"Database connectivity check failed: {exc}"
        ) from exc


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise TrackerConfigError("Engine not initialized. Call init_tracker() first.")
    return _engine


async def dispose_engine() -> None:
    """Close the connection pool and clear the global engine.

    Call from your ASGI shutdown handler (after ``wait_pending_writes``) to
    release Postgres connections cleanly. Also useful in tests and short-lived
    workers where leaving a pool dangling delays process exit.

    Safe to call when the engine has not been initialised — it becomes a
    no-op rather than raising.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


@contextlib.asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    if _session_factory is None:
        raise TrackerConfigError("Session factory not initialized. Call init_tracker() first.")
    async with _session_factory() as session:
        yield session
