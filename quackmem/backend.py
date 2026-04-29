from __future__ import annotations

import logging
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import insert, update, select
from sqlalchemy.exc import OperationalError
from tenacity import retry, retry_if_exception, wait_random_exponential, stop_after_attempt

from quackmem.db import get_session, tracked_sessions, tracked_messages
from quackmem.schema.models import TrackedSession, TrackedMessage
from quackmem.schema.enums import MessageStatus

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


def _session_values(session: TrackedSession) -> dict:
    d = session.model_dump()
    return d


def _message_values(message: TrackedMessage) -> dict:
    d = message.model_dump()
    return d


class PostgresBackend:

    @_retryable
    async def create_session(self, session: TrackedSession) -> None:
        """Create session row. Idempotent — silently ignores duplicate id."""
        async with get_session() as db:
            stmt = insert(tracked_sessions).values(**_session_values(session))
            stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
            await db.execute(stmt)
            await db.commit()

    @_retryable
    async def insert_message(self, message: TrackedMessage) -> None:
        """Insert a message row atomically."""
        async with get_session() as db:
            await db.execute(insert(tracked_messages).values(**_message_values(message)))
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
        Must not create new rows — only updates existing ones."""
        from datetime import datetime, UTC
        async with get_session() as db:
            values: dict = {
                "content": content,
                "updated_at": datetime.now(UTC),
            }
            if regeneration_count is not None:
                values["regeneration_count"] = regeneration_count
            await db.execute(
                update(tracked_messages)
                .where(tracked_messages.c.id == message_id)
                .values(**values)
            )
            await db.commit()

    async def get_messages(self, session_id: UUID) -> list[dict]:
        """Return all messages for a session ordered by created_at ascending."""
        async with get_session() as db:
            result = await db.execute(
                select(tracked_messages)
                .where(tracked_messages.c.session_id == session_id)
                .order_by(tracked_messages.c.created_at.asc())
            )
            return [dict(row._mapping) for row in result]

    async def update_status(self, message_id: UUID, status: MessageStatus) -> None:
        """Update only the status column of a message row."""
        stmt = sa.update(tracked_messages).where(tracked_messages.c.id == message_id).values(status=status.value)
        async with get_session() as db_session:
            await db_session.execute(stmt)
            await db_session.commit()


_backend: PostgresBackend | None = None


def get_backend() -> PostgresBackend:
    global _backend
    if _backend is None:
        _backend = PostgresBackend()
    return _backend
