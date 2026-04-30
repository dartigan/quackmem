from __future__ import annotations

import logging
from datetime import timedelta
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import bindparam, update, select, cast
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.exc import OperationalError
from tenacity import retry, retry_if_exception, wait_random_exponential, stop_after_attempt

from quackmem.db import get_session
from quackmem.db import tables as _tables
from quackmem.schema.models import (
    TrackedSession,
    TrackedMessage,
    MessageReservation,
    MessageFinalization,
)
from quackmem.schema.enums import MessageStatus


def _sessions_table():
    """Resolve the tracked_sessions Table at call time. ``init_tracker`` builds
    the Table object lazily, so capturing it at import time would freeze it
    to ``None`` for any caller that imports backend before init."""
    if _tables.tracked_sessions is None:
        raise RuntimeError(
            "Tables not initialised. Call init_tracker() before using the backend."
        )
    return _tables.tracked_sessions


def _messages_table():
    if _tables.tracked_messages is None:
        raise RuntimeError(
            "Tables not initialised. Call init_tracker() before using the backend."
        )
    return _tables.tracked_messages

logger = logging.getLogger(__name__)

RETRYABLE_PG_CODES = {
    "40001",  # serialization failure
    "40P01",  # deadlock detected
    "08000",  # connection exception
    "08003",  # connection does not exist
    "08006",  # connection failure
    "40000",  # transaction rollback
}


def is_retryable_sqlalchemy_error(exc: Exception) -> bool:
    if not isinstance(exc, OperationalError):
        return False
    orig = getattr(exc, "orig", None)
    pgcode = getattr(orig, "pgcode", None)
    return pgcode in RETRYABLE_PG_CODES


def _retryable(fn):
    """Decorator: retry on transient Postgres errors with exponential backoff + jitter."""
    return retry(
        retry=retry_if_exception(is_retryable_sqlalchemy_error),
        wait=wait_random_exponential(multiplier=0.1, max=1),
        stop=stop_after_attempt(3),
        reraise=True,
    )(fn)


class PostgresBackend:

    @_retryable
    async def create_session(self, session: TrackedSession) -> None:
        """Create session row. Idempotent — silently ignores duplicate id."""
        sessions = _sessions_table()
        async with get_session() as db:
            stmt = insert(sessions).values(**session.model_dump())
            stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
            await db.execute(stmt)
            await db.commit()

    @_retryable
    async def insert_message(self, message: TrackedMessage) -> None:
        """Insert a message row atomically."""
        messages = _messages_table()
        async with get_session() as db:
            await db.execute(insert(messages).values(**message.model_dump()))
            await db.commit()

    @_retryable
    async def update_message(
        self,
        message_id: UUID,
        content: str,
        regeneration_count: int | None = None,
    ) -> None:
        """Update an existing assistant message. Used for regeneration.
        Updates content, updated_at, and optionally regeneration_count.
        When regeneration_count is None, auto-increments the existing value.
        Must not create new rows — only updates existing ones."""
        from datetime import datetime, UTC
        messages = _messages_table()
        async with get_session() as db:
            values: dict = {
                "content": content,
                "updated_at": datetime.now(UTC),
            }
            if regeneration_count is not None:
                values["regeneration_count"] = regeneration_count
            else:
                values["regeneration_count"] = messages.c.regeneration_count + 1
            await db.execute(
                update(messages)
                .where(messages.c.id == message_id)
                .values(**values)
            )
            await db.commit()

    @_retryable
    async def reserve_assistant_message(
        self, reservation: MessageReservation
    ) -> MessageReservation:
        """Insert an empty assistant message with status=pending.

        Called once before the wrapped function runs so the caller can observe
        a stable message_id mid-call. Returns the reservation (with its
        assigned id) for chaining."""
        messages = _messages_table()
        async with get_session() as db:
            await db.execute(
                insert(messages).values(
                    id=reservation.id,
                    session_id=reservation.session_id,
                    conversation_id=reservation.conversation_id,
                    parent_message_id=reservation.parent_message_id,
                    role=reservation.role,
                    content="",
                    tool_call_id=reservation.tool_call_id,
                    token_count=None,
                    status=MessageStatus.pending.value,
                    metadata=reservation.metadata,
                )
            )
            await db.commit()
        return reservation

    @_retryable
    async def finalize_message(self, finalization: MessageFinalization) -> None:
        """Finalize a previously reserved message with content + status.

        Called exactly once per reservation. On error paths, the caller passes
        ``status=failed`` and an ``error`` string which is merged into the
        existing metadata so the audit trail is preserved.

        The UPDATE is gated on ``status='pending'`` so a late finalize cannot
        overwrite a row that the orphan reaper has already moved to ``failed``,
        and a duplicate finalize from a retry won't clobber a completed row.
        Mismatches are logged + counted; they don't raise.
        """
        from datetime import datetime, UTC
        from quackmem.core import metrics
        messages = _messages_table()
        async with get_session() as db:
            values: dict = {
                "content": finalization.content,
                "status": finalization.status,
                "updated_at": datetime.now(UTC),
            }
            if finalization.tool_calls is not None:
                values["tool_calls"] = finalization.tool_calls
            if finalization.token_count is not None:
                values["token_count"] = finalization.token_count
            if finalization.error is not None:
                row = (await db.execute(
                    select(messages.c.metadata).where(
                        messages.c.id == finalization.message_id
                    )
                )).first()
                current_meta = dict(row[0]) if row and row[0] else {}
                current_meta["error"] = finalization.error
                values["metadata"] = current_meta
            result = await db.execute(
                update(messages)
                .where(messages.c.id == finalization.message_id)
                .where(messages.c.status == MessageStatus.pending.value)
                .values(**values)
            )
            await db.commit()
            if (result.rowcount or 0) == 0:  # type: ignore[attr-defined]
                metrics.increment("finalize_status_mismatch")
                logger.warning(
                    "finalize_message: no pending row matched; reaper or "
                    "duplicate finalize likely won the race",
                    extra={"message_id": str(finalization.message_id)},
                )

    async def get_messages(self, session_id: UUID) -> list[dict]:
        """Return all messages for a session ordered by created_at ascending.

        Internal use only — backs the decorator's history-dedup pass. Public
        consumers should call :meth:`read_messages` instead so they pick up
        status filtering and the default-limit behaviour.
        """
        messages = _messages_table()
        async with get_session() as db:
            result = await db.execute(
                select(messages)
                .where(messages.c.session_id == session_id)
                .order_by(messages.c.created_at.asc())
            )
            return [dict(row._mapping) for row in result]

    async def read_messages(
        self,
        session_id: UUID,
        *,
        limit: int,
        statuses: set[MessageStatus],
    ) -> list[dict]:
        """Return the most recent ``limit`` messages for ``session_id``.

        Rows are selected ``ORDER BY created_at DESC, id DESC`` so the index
        scan picks the newest first; the result is then reversed in Python to
        chronological order, which is what LLM-history consumers want.

        Args:
            session_id: Session to read.
            limit: Maximum number of messages to return. The caller is
                expected to apply ``TrackerConfig.default_read_limit`` when
                they don't have an explicit limit.
            statuses: Only rows whose ``status`` is in this set are returned.
                Pass ``{MessageStatus.completed}`` for normal history reads.

        Returns:
            Messages in chronological (oldest-first) order. May be shorter
            than ``limit`` (or empty) when the session has fewer matching
            rows.
        """
        messages = _messages_table()
        status_values = [s.value for s in statuses]
        async with get_session() as db:
            result = await db.execute(
                select(messages)
                .where(messages.c.session_id == session_id)
                .where(messages.c.status.in_(status_values))
                .order_by(messages.c.created_at.desc(), messages.c.id.desc())
                .limit(limit)
            )
            rows = [dict(row._mapping) for row in result]
        rows.reverse()
        return rows

    @_retryable
    async def reap_orphans(self, older_than: timedelta) -> int:
        """Mark long-running ``pending`` rows as ``failed`` with error="orphaned".

        A row stays ``pending`` only between ``reserve_assistant_message`` and
        ``finalize_message``. If the host process dies in that window — or
        ``finalize_message`` itself exhausts its retries — the row is stuck.
        Call this periodically (cron, scheduler, startup hook) to clear them.

        Args:
            older_than: Only rows whose ``created_at`` is older than
                ``now() - older_than`` are reaped. Use this to avoid touching
                in-flight rows from concurrent workers.

        Returns:
            Number of rows reaped.
        """
        from datetime import datetime, UTC
        messages = _messages_table()
        cutoff = datetime.now(UTC) - older_than
        # Postgres ``||`` on jsonb merges right-side keys over left.
        merged_meta = messages.c.metadata.op("||")(
            cast(sa.literal('{"error": "orphaned"}'), JSONB)
        )
        async with get_session() as db:
            result = await db.execute(
                update(messages)
                .where(messages.c.status == MessageStatus.pending.value)
                .where(messages.c.created_at < cutoff)
                .values(
                    status=MessageStatus.failed.value,
                    updated_at=datetime.now(UTC),
                    metadata=merged_meta,
                )
            )
            await db.commit()
            from quackmem.core import metrics
            reaped = result.rowcount or 0  # type: ignore[attr-defined]
            if reaped:
                metrics.increment("reservations_unfinalized", reaped)
            return reaped

    @_retryable
    async def insert_tool_result(
        self,
        *,
        session_id: UUID,
        conversation_id: UUID,
        tool_call_id: str,
        content: str | list[dict],
        parent_message_id: UUID | None = None,
        metadata: dict | None = None,
    ) -> UUID:
        """Insert a ``role=tool`` row that answers a previous tool call.

        Returns the inserted message id. ``parent_message_id`` should point
        at the assistant row that issued the call when known; pass ``None``
        if the lookup was inconclusive — quackmem prefers to record the
        result with a NULL link over losing it.
        """
        from uuid import uuid4
        messages = _messages_table()
        new_id = uuid4()
        async with get_session() as db:
            await db.execute(
                insert(messages).values(
                    id=new_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    parent_message_id=parent_message_id,
                    role="tool",
                    content=content,
                    tool_call_id=tool_call_id,
                    status=MessageStatus.completed.value,
                    metadata=metadata or {},
                )
            )
            await db.commit()
        return new_id

    async def get_conversation_id(self, session_id: UUID) -> UUID | None:
        """Return the ``conversation_id`` for a session, or ``None`` if the
        session row doesn't exist. Used by ``record_tool_result`` so callers
        don't have to pass ``conversation_id`` redundantly."""
        sessions = _sessions_table()
        async with get_session() as db:
            row = (await db.execute(
                select(sessions.c.conversation_id).where(sessions.c.id == session_id)
            )).first()
        return UUID(str(row[0])) if row else None

    async def find_assistant_for_tool_call(
        self, session_id: UUID, tool_call_id: str
    ) -> UUID | None:
        """Return the most recent assistant message in this session whose
        ``tool_calls`` array contains an entry with the given id, or ``None``
        if no such row exists. Used by ``record_tool_result`` to back-link
        the result row to its caller."""
        messages = _messages_table()
        # Postgres jsonb @> contains-any-element check. Bind the needle as a
        # parameter so user-supplied tool_call_id can't break the JSON literal
        # or escape the SQL.
        needle = bindparam(
            "tool_call_needle",
            value=[{"id": tool_call_id}],
            type_=JSONB,
        )
        async with get_session() as db:
            result = await db.execute(
                select(messages.c.id)
                .where(messages.c.session_id == session_id)
                .where(messages.c.role == "assistant")
                .where(messages.c.tool_calls.is_not(None))
                .where(messages.c.tool_calls.op("@>")(needle))
                .order_by(messages.c.created_at.desc(), messages.c.id.desc())
                .limit(1)
            )
            row = result.first()
            return UUID(str(row[0])) if row else None

    async def update_status(self, message_id: UUID, status: MessageStatus) -> None:
        """Update only the status column of a message row."""
        messages = _messages_table()
        stmt = sa.update(messages).where(messages.c.id == message_id).values(status=status.value)
        async with get_session() as db_session:
            await db_session.execute(stmt)
            await db_session.commit()


_backend: PostgresBackend | None = None


def get_backend() -> PostgresBackend:
    global _backend
    if _backend is None:
        _backend = PostgresBackend()
    return _backend
