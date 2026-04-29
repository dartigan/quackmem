"""convo-tracker: Persist AI agent conversations to Postgres. One decorator, zero opinions."""
from __future__ import annotations

from convo_tracker.core.config import TrackerConfig
from convo_tracker.core.registry import register_metadata
from convo_tracker.core.context import get_tracking_context
from convo_tracker.core.exceptions import TrackerConfigError, MetadataValidationError
from convo_tracker.migrations.runner import upgrade_db, downgrade_db


def init_tracker(config: TrackerConfig) -> None:
    """Initialize convo-tracker. Call once at application startup, after upgrade_db()."""
    from convo_tracker.db.session import init_engine
    from convo_tracker.db.tables import build_tables

    if not config.database_url:
        raise TrackerConfigError("TrackerConfig.database_url is required")

    # URL auto-correction (postgresql:// -> postgresql+asyncpg://) is handled in
    # TrackerConfig validation — see convo_tracker/core/config.py.
    init_engine(config)

    schema = config.schema_name if config.schema_name != "public" else None
    build_tables(schema=schema, prefix=config.table_prefix)


__all__ = [
    "TrackerConfig",
    "init_tracker",
    "register_metadata",
    "get_tracking_context",
    "upgrade_db",
    "downgrade_db",
]
