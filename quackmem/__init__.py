"""quackmem: Persist AI agent conversations to Postgres. One decorator, zero opinions."""
from __future__ import annotations

from quackmem.core.config import TrackerConfig
from quackmem.core.registry import register_metadata, reset_metadata
from quackmem.core.context import get_tracking_context
from quackmem.core.exceptions import TrackerConfigError, MetadataValidationError
from quackmem.migrations.runner import upgrade_db, downgrade_db, generate_migration


def init_tracker(config: TrackerConfig) -> None:
    """Initialize quackmem. Call once at application startup, after upgrade_db()."""
    from quackmem.db.session import init_engine
    from quackmem.db.tables import build_tables

    init_engine(config)

    schema = config.schema_name if config.schema_name != "public" else None
    build_tables(schema=schema, prefix=config.table_prefix)


async def verify_tracker() -> None:
    """Verify the database is reachable. Call after init_tracker() at startup.

    Raises:
        TrackerConfigError: If init_tracker() has not been called or the
            database cannot be reached.

    Example::

        config = TrackerConfig(database_url="postgresql://...")
        upgrade_db(database_url=config.database_url)
        init_tracker(config)
        await verify_tracker()  # raises immediately if DB is unreachable
    """
    from quackmem.db.session import verify_engine
    await verify_engine()


__all__ = [
    "TrackerConfig",
    "TrackerConfigError",
    "MetadataValidationError",
    "init_tracker",
    "verify_tracker",
    "register_metadata",
    "reset_metadata",
    "get_tracking_context",
    "upgrade_db",
    "downgrade_db",
    "generate_migration",
]
