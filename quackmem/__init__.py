"""quackmem: Persist AI agent conversations to Postgres. One decorator, zero opinions."""
from __future__ import annotations

from datetime import timedelta  # noqa: F401  (used in annotation strings)

from quackmem.core.config import TrackerConfig
from quackmem.core.registry import register_metadata, reset_metadata
from quackmem.core.context import get_tracking_context
from quackmem.core.exceptions import TrackerConfigError, MetadataValidationError
from quackmem.core.logging import configure_logging
from quackmem.core.tasks import wait_pending_writes
from quackmem.db.session import dispose_engine
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


async def reap_orphans(older_than: "timedelta | None" = None) -> int:
    """Mark stuck ``pending`` messages as ``failed`` with error="orphaned".

    A reserved assistant row stays ``pending`` between pre-write and post-write.
    If the host process dies in that window, the row is orphaned. Call this at
    startup (after :func:`init_tracker`) and/or wire it into your own scheduler
    (cron, APScheduler, Celery beat) on the cadence that fits your app — the
    library does not run a background sweeper itself.

    Args:
        older_than: Only rows older than ``now() - older_than`` are reaped.
            Defaults to 10 minutes, which is well past any reasonable LLM call
            duration. Pass a smaller value at your own risk: in-flight rows
            from concurrent workers can be incorrectly marked failed.

    Returns:
        Number of rows reaped.
    """
    from datetime import timedelta
    from quackmem.backend import get_backend

    if older_than is None:
        older_than = timedelta(minutes=10)
    return await get_backend().reap_orphans(older_than)


async def shutdown_tracker(*, drain_timeout: float | None = 10) -> int:
    """Drain pending tracking writes and close the connection pool.

    Convenience wrapper around :func:`wait_pending_writes` followed by
    :func:`quackmem.db.dispose_engine`. Use this in your ASGI shutdown handler
    (or any long-running worker exit path) so messages aren't lost and
    Postgres connections are returned to the OS promptly.

    Args:
        drain_timeout: Seconds to wait for in-flight tracking writes. ``None``
            waits indefinitely. Defaults to 10s.

    Returns:
        Number of tracking writes still in flight when the timeout elapsed
        (``0`` on a clean shutdown).
    """
    from quackmem.db.session import dispose_engine
    pending = await wait_pending_writes(timeout=drain_timeout)
    await dispose_engine()
    return pending


__all__ = [
    "TrackerConfig",
    "TrackerConfigError",
    "MetadataValidationError",
    "init_tracker",
    "verify_tracker",
    "reap_orphans",
    "shutdown_tracker",
    "register_metadata",
    "reset_metadata",
    "get_tracking_context",
    "wait_pending_writes",
    "dispose_engine",
    "configure_logging",
    "upgrade_db",
    "downgrade_db",
    "generate_migration",
]
