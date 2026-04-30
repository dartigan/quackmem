"""quackmem: Persist AI agent conversations to Postgres. One decorator, zero opinions."""
from __future__ import annotations

from datetime import timedelta  # noqa: F401  (used in annotation strings)
from typing import TYPE_CHECKING

from quackmem.core.config import TrackerConfig
from quackmem.schema.canonical import ToolCall
from quackmem.core.registry import register_metadata, reset_metadata
from quackmem.core.context import get_tracking_context
from quackmem.core.exceptions import TrackerConfigError, MetadataValidationError
from quackmem.core.logging import configure_logging
from quackmem.core.tasks import wait_pending_writes
from quackmem.db.session import dispose_engine
from quackmem.migrations.runner import upgrade_db, downgrade_db, generate_migration

if TYPE_CHECKING:
    from quackmem.schema.enums import MessageStatus
    from quackmem.schema.models import TrackedMessage


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


async def read_messages(
    session_id,
    *,
    limit: int | None = None,
    statuses: "set[MessageStatus] | None" = None,
) -> "list[TrackedMessage]":
    """Read recent messages for a session.

    Returns the most recent ``limit`` messages for ``session_id`` in
    chronological (oldest-first) order — the shape LLM history consumers
    expect. Backed by a composite index on ``(session_id, created_at, id)``.

    Args:
        session_id: Session UUID. Returning empty when the session is unknown
            or has no matching messages.
        limit: Maximum number of messages. Defaults to
            ``TrackerConfig.default_read_limit`` (env
            ``QUACKMEM_DEFAULT_READ_LIMIT``, default ``10``). Pass an explicit
            integer to override on a per-call basis.
        statuses: Status filter. ``None`` means ``{MessageStatus.completed}``
            — ``pending`` rows have empty content and ``failed`` rows hold
            stale text from a crashed turn, so neither belongs in normal
            history reads. Pass an explicit set to opt into them (e.g. for
            debugging tools).

    Returns:
        Validated :class:`TrackedMessage` instances in chronological order.
        May be shorter than ``limit`` (or empty) when fewer rows match.
    """
    from uuid import UUID

    from quackmem.backend import get_backend
    from quackmem.db.session import get_config
    from quackmem.schema.enums import MessageStatus as _Status
    from quackmem.schema.models import TrackedMessage as _TrackedMessage

    if not isinstance(session_id, UUID):
        session_id = UUID(str(session_id))
    if statuses is None:
        statuses = {_Status.completed}
    if limit is None:
        limit = get_config().default_read_limit

    rows = await get_backend().read_messages(
        session_id, limit=limit, statuses=statuses,
    )
    return [_TrackedMessage(**row) for row in rows]


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


async def record_tool_result(
    session_id,
    tool_call_id: str,
    content,
    *,
    parent_message_id=None,
    conversation_id=None,
    name: str | None = None,
    metadata: dict | None = None,
):
    """Persist a ``role=tool`` row that answers a previous tool call.

    Use this from inside your tool-execution loop to record what the tool
    returned, so the conversation can be replayed end-to-end. One call per
    tool result; quackmem does not auto-detect them.

    Args:
        session_id: Session UUID (or any value coercible to UUID).
        tool_call_id: The id of the call this row answers — must match an
            entry in the assistant message's ``tool_calls`` array. Used both
            as the linkage key and to look up ``parent_message_id`` when not
            supplied.
        content: The tool's output. Either a string or a list-of-dicts (for
            structured returns).
        parent_message_id: The assistant message that issued the call. If
            omitted, quackmem looks up the most recent assistant row in this
            session whose ``tool_calls`` array contains ``tool_call_id``.
            If the lookup misses, the row is still inserted with a NULL
            link rather than failing.
        conversation_id: If omitted, derived from the session row.
        name: Optional tool/function name. Stored under ``metadata.tool_name``
            for convenience.
        metadata: Free-form metadata merged onto the row.

    Returns:
        UUID of the inserted message.

    Raises:
        TrackerConfigError: If ``init_tracker`` has not been called.
        SQLAlchemyError: On unrecoverable DB errors. Unlike the decorator,
            this helper does not silently swallow failures — the caller
            took deliberate action and should see the error.
    """
    from uuid import UUID

    from quackmem.backend import get_backend

    if not isinstance(session_id, UUID):
        session_id = UUID(str(session_id))

    backend = get_backend()

    if parent_message_id is None:
        parent_message_id = await backend.find_assistant_for_tool_call(
            session_id, tool_call_id
        )
    elif not isinstance(parent_message_id, UUID):
        parent_message_id = UUID(str(parent_message_id))

    if conversation_id is None:
        conversation_id = await backend.get_conversation_id(session_id)
        if conversation_id is None:
            raise ValueError(
                f"Unknown session_id={session_id}. Call the persona/decorator at "
                "least once before recording tool results, or pass conversation_id "
                "explicitly."
            )
    elif not isinstance(conversation_id, UUID):
        conversation_id = UUID(str(conversation_id))

    merged_meta = dict(metadata or {})
    if name is not None:
        merged_meta.setdefault("tool_name", name)

    return await backend.insert_tool_result(
        session_id=session_id,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        content=content,
        parent_message_id=parent_message_id,
        metadata=merged_meta,
    )


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
    "ToolCall",
    "TrackerConfig",
    "TrackerConfigError",
    "MetadataValidationError",
    "init_tracker",
    "verify_tracker",
    "reap_orphans",
    "read_messages",
    "record_tool_result",
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
