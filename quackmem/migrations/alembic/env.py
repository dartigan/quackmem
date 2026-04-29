from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import the library's metadata so autogenerate can diff against it.
# build_tables() must be called before autogenerate runs so the Table
# objects are registered on the metadata object.
try:
    from quackmem.db.tables import metadata as target_metadata, build_tables
    # Build tables with default schema so metadata is populated
    build_tables(schema=None, prefix="")
except Exception:
    target_metadata = None


def get_url() -> str:
    """Get database URL from environment or initialized engine.

    Returns:
        Database URL string

    Raises:
        RuntimeError: If no database URL is available
    """
    url = os.environ.get("QUACKMEM_DB_URL")
    if not url:
        try:
            from quackmem.db.session import get_engine

            engine = get_engine()
            url = str(engine.url)
        except Exception:
            raise RuntimeError(
                "Set QUACKMEM_DB_URL or call init_tracker() before running migrations"
            )
    return url


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
