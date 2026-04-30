from __future__ import annotations

import contextlib
import re
from collections.abc import AsyncIterator

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from quackmem.core.config import TrackerConfig
from quackmem.core.exceptions import TrackerConfigError

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None
_config: TrackerConfig | None = None


def get_config() -> TrackerConfig:
    """Return the active TrackerConfig.

    Raises:
        TrackerConfigError: If ``init_engine`` / ``init_tracker`` has not
            been called.
    """
    if _config is None:
        raise TrackerConfigError("Config not initialized. Call init_tracker() first.")
    return _config


def init_engine(config: TrackerConfig) -> None:
    global _engine, _session_factory, _config
    _config = config
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
            f"Database connectivity check failed: {_mask_dsn(str(exc))}"
        ) from exc


# Matches the userinfo segment of a URL: ``scheme://user:password@host``.
# Drivers occasionally render the full DSN (with password) into exception
# messages; strip the password before re-raising so it doesn't end up in
# logs / Sentry.
_DSN_RE = re.compile(r"(://[^:/@\s]+):([^@/\s]+)@")


def _mask_dsn(text: str) -> str:
    return _DSN_RE.sub(r"\1:***@", text)


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
    global _engine, _session_factory, _config
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
    _config = None


@contextlib.asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    if _session_factory is None:
        raise TrackerConfigError("Session factory not initialized. Call init_tracker() first.")
    async with _session_factory() as session:
        yield session
