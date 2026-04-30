from __future__ import annotations

import asyncio
import logging
import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from quackmem.core.exceptions import TrackerConfigError

logger = logging.getLogger(__name__)

config = context.config
# Only configure logging from the ini if it actually contains logging sections.
# The bundled alembic.ini deliberately omits them so quackmem doesn't fight the
# host application's logging setup.
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except KeyError:
        pass

# Import the library's metadata so autogenerate can diff against it.
# build_tables() must be called before autogenerate runs so the Table
# objects are registered on the metadata object.
target_metadata: Any = None
try:
    from quackmem.db.tables import metadata as target_metadata, build_tables  # noqa: F811
    # Build tables with default schema so metadata is populated
    build_tables(schema=None, prefix="")
except (ImportError, RuntimeError) as exc:
    # ImportError: quackmem isn't installed (autogenerate from outside the
    # package). RuntimeError: build_tables called twice with conflicting args.
    # In either case autogenerate will compare against an empty metadata,
    # which is benign — but log loudly so the operator notices.
    logger.warning(
        "quackmem table metadata unavailable; autogenerate will be a no-op",
        exc_info=exc,
    )
    target_metadata = None


def get_url() -> str:
    """Get database URL from Alembic config, environment, or initialized engine.

    Resolution order:
    1. ``sqlalchemy.url`` injected into the Alembic config by ``_make_alembic_config``
       (set via the ``database_url`` parameter of ``upgrade_db`` / ``downgrade_db``).
    2. ``QUACKMEM_DB_URL`` environment variable.
    3. An already-initialised SQLAlchemy engine (requires ``init_tracker()`` to have
       been called before running migrations).

    Returns:
        Database URL string

    Raises:
        RuntimeError: If no database URL is available from any source
    """
    # 1. URL injected directly into the config by _make_alembic_config
    url = config.get_section_option("alembic", "sqlalchemy.url")
    if url:
        return url

    # 2. Environment variable fallback
    url = os.environ.get("QUACKMEM_DB_URL")
    if url:
        return url

    # 3. Already-initialised engine
    try:
        from quackmem.db.session import get_engine

        engine = get_engine()
        return str(engine.url)
    except TrackerConfigError:
        # Engine simply hasn't been initialised yet — fall through to the
        # final RuntimeError so the caller sees a clear message.
        pass

    raise RuntimeError(
        "No database URL available. Pass database_url to upgrade_db(), "
        "set QUACKMEM_DB_URL, or call init_tracker() before running migrations."
    )


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""
    schema = config.get_section_option("alembic", "target_schema") or "public"
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="quackmem_alembic_version",
        version_table_schema=schema,
        include_schemas=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Run migrations with the given connection."""
    schema = config.get_section_option("alembic", "target_schema") or "public"
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table="quackmem_alembic_version",
        version_table_schema=schema,
        include_schemas=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations asynchronously."""
    engine = create_async_engine(get_url())
    async with engine.begin() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    """Run migrations in online mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
