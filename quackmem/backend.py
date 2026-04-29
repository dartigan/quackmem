from __future__ import annotations

import logging
from uuid import UUID

import sqlalchemy as sa

from quackmem.db import get_session, tracked_sessions, tracked_messages
from quackmem.schema.models import TrackedSession, TrackedMessage
from quackmem.schema.enums import MessageStatus

logger = logging.getLogger(__name__)


class PostgresBackend:

    async def upsert_session(self, session: TrackedSession) -> None:
        """Insert session row. Silently ignores conflicts on id (idempotent)."""
        stmt = sa.insert(tracked_sessions).values(session.model_dump()).on_conflict_do_nothing(index_elements=["id"])
        async with get_session() as db_session:
            await db_session.execute(stmt)
            await db_session.commit()

    async def insert_message(self, message: TrackedMessage) -> None:
        """Insert a message row."""
        stmt = sa.insert(tracked_messages).values(message.model_dump())
        async with get_session() as db_session:
            await db_session.execute(stmt)
            await db_session.commit()

    async def update_status(self, message_id: UUID, status: MessageStatus) -> None:
        """Update only the status column of a message row."""
        stmt = sa.update(tracked_messages).where(tracked_messages.c.id == message_id).values(status=status.value)
        async with get_session() as db_session:
            await db_session.execute(stmt)
            await db_session.commit()

    async def get_messages(self, session_id: UUID) -> list[dict]:
        """Return all messages for a session ordered by created_at ascending."""
        stmt = sa.select(tracked_messages).where(tracked_messages.c.session_id == session_id).order_by(tracked_messages.c.created_at.asc())
        async with get_session() as db_session:
            result = await db_session.execute(stmt)
            return [dict(row._mapping) for row in result]


_backend: PostgresBackend | None = None


def get_backend() -> PostgresBackend:
    global _backend
    if _backend is None:
        _backend = PostgresBackend()
    return _backend
