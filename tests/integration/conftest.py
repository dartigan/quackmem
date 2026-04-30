"""Fixtures for integration tests that hit a real Postgres.

Two backends are supported, in priority order:

1. ``QUACKMEM_TEST_DB_URL`` env var — points at an existing Postgres. CI
   typically uses a service container and exports this variable. The fixture
   drops/recreates the schema between tests for isolation.
2. ``pytest-postgresql`` — spawns a temporary Postgres process. Requires
   the Postgres server binaries (``pg_ctl``) to be on PATH.

If neither is available, all integration tests are skipped.
"""
from __future__ import annotations

import os
import shutil

import pytest
import pytest_asyncio


def _have_pg_binaries() -> bool:
    return shutil.which("pg_ctl") is not None


# Per-test ephemeral DB from pytest-postgresql when available. The fixture
# is parameterised at collection time so it doesn't fire when the user has
# pre-set QUACKMEM_TEST_DB_URL or doesn't have pg_ctl.
try:
    from pytest_postgresql import factories  # noqa: F401
    _PYTEST_POSTGRESQL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYTEST_POSTGRESQL_AVAILABLE = False


@pytest_asyncio.fixture
async def db_url(request) -> str:
    """Yield a fresh, migrated Postgres URL for the test, then tear it down.

    Skips the test if no real Postgres is reachable.
    """
    explicit = os.environ.get("QUACKMEM_TEST_DB_URL")
    if explicit:
        url = _normalise_url(explicit)
        await _reset_schema(url)
        yield url
        await _reset_schema(url)
        return

    if not (_PYTEST_POSTGRESQL_AVAILABLE and _have_pg_binaries()):
        pytest.skip(
            "Integration tests need either QUACKMEM_TEST_DB_URL or "
            "pytest-postgresql + pg_ctl on PATH"
        )

    pg = request.getfixturevalue("postgresql")
    info = pg.info
    url = (
        f"postgresql+asyncpg://{info.user}:@{info.host}:{info.port}/{info.dbname}"
    )
    yield url


def _normalise_url(url: str) -> str:
    url = url.strip()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


async def _reset_schema(url: str) -> None:
    """Drop and recreate the public schema so tests are isolated when sharing
    a long-lived Postgres."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text

    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def initialized_tracker(db_url: str):
    """Run migrations + init_tracker against the supplied DB; reset on teardown."""
    import asyncio

    from quackmem import init_tracker, upgrade_db, verify_tracker
    from quackmem.core.config import TrackerConfig
    from quackmem.db import dispose_engine

    # Make sure no leftover engine from a previous test interferes.
    await dispose_engine()

    # upgrade_db() calls asyncio.run() internally, which conflicts with the
    # already-running pytest-asyncio loop. Run it in a thread.
    await asyncio.to_thread(upgrade_db, database_url=db_url)
    init_tracker(TrackerConfig(database_url=db_url))
    await verify_tracker()

    yield db_url

    await dispose_engine()
