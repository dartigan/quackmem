"""quackmem: Persist AI agent conversations to Postgres. One decorator, zero opinions."""
from __future__ import annotations

from quackmem.core.config import TrackerConfig
from quackmem.core.registry import register_metadata
from quackmem.core.context import get_tracking_context
from quackmem.core.exceptions import TrackerConfigError, MetadataValidationError
from quackmem.migrations.runner import upgrade_db, downgrade_db


def init_tracker(config: TrackerConfig) -> None:
    """Initialize quackmem. Call once at application startup, after upgrade_db()."""
    from quackmem.db.session import init_engine
    from quackmem.db.tables import build_tables

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
