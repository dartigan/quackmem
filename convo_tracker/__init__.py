"""convo-tracker: Persist AI agent conversations to Postgres. One decorator, zero opinions."""
from __future__ import annotations

from convo_tracker.core.config import TrackerConfig
from convo_tracker.core.registry import register_metadata
from convo_tracker.core.context import get_tracking_context
from convo_tracker.core.exceptions import TrackerConfigError, MetadataValidationError
from convo_tracker.migrations.runner import upgrade_db, downgrade_db


def init_tracker(config: TrackerConfig) -> None:
    """
    Initialize convo-tracker. Call once at application startup, after upgrade_db().

    This initializes:
    - The async SQLAlchemy engine and connection pool
    - The SQLAlchemy table objects with the correct schema and prefix
    - The Celery app (skipped when config.sync_mode is True)
    - The decorator's reference to config (for sync_mode flag)
    """
    from convo_tracker.db.session import init_engine
    from convo_tracker.db.tables import build_tables
    from convo_tracker.core.decorator import init_decorator

    # Validate database URL
    if not config.database_url:
        raise TrackerConfigError("TrackerConfig.database_url is required")

    # Ensure URL uses asyncpg driver
    db_url = config.database_url
    if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1).replace(
            "postgres://", "postgresql+asyncpg://", 1
        )
        # Create a new config with corrected URL
        import dataclasses

        config = dataclasses.replace(config, database_url=db_url)

    init_engine(config)

    schema = config.schema_name if config.schema_name != "public" else None
    build_tables(schema=schema, prefix=config.table_prefix)

    if not config.sync_mode:
        from convo_tracker.worker.celery_app import init_celery

        try:
            init_celery(config.celery_broker_url, config.celery_result_backend)
        except Exception as exc:
            raise TrackerConfigError(f"Failed to initialize Celery: {exc}") from exc

    init_decorator(config)


__all__ = [
    "TrackerConfig",
    "init_tracker",
    "register_metadata",
    "get_tracking_context",
    "upgrade_db",
    "downgrade_db",
]
