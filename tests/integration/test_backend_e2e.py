"""Direct backend integration tests against a real Postgres."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from quackmem.schema.enums import MessageRole, MessageStatus
from quackmem.schema.models import TrackedMessage, TrackedSession


def _make_session(**overrides) -> TrackedSession:
    defaults = dict(conversation_id=uuid4())
    defaults.update(overrides)
    return TrackedSession(**defaults)


def _make_message(session: TrackedSession, **overrides) -> TrackedMessage:
    defaults = dict(
        session_id=session.id,
        conversation_id=session.conversation_id,
        role=MessageRole.assistant,
        content="test content",
        status=MessageStatus.completed,
    )
    defaults.update(overrides)
    return TrackedMessage(**defaults)


@pytest.mark.asyncio
async def test_create_session_idempotent(initialized_tracker):
    from quackmem.backend import get_backend

    backend = get_backend()
    session = _make_session()
    await backend.create_session(session)
    # Second call with the same id must not raise (on_conflict_do_nothing).
    await backend.create_session(session)


@pytest.mark.asyncio
async def test_insert_and_get_messages(initialized_tracker):
    from quackmem.backend import get_backend

    backend = get_backend()
    session = _make_session()
    message = _make_message(session)

    await backend.create_session(session)
    await backend.insert_message(message)
    rows = await backend.get_messages(session.id)

    assert len(rows) == 1
    assert str(rows[0]["id"]) == str(message.id)


@pytest.mark.asyncio
async def test_update_status_only_changes_status(initialized_tracker):
    from quackmem.backend import get_backend

    backend = get_backend()
    session = _make_session()
    message = _make_message(session, status=MessageStatus.pending)

    await backend.create_session(session)
    await backend.insert_message(message)
    await backend.update_status(message.id, MessageStatus.completed)

    rows = await backend.get_messages(session.id)
    assert rows[0]["status"] == MessageStatus.completed.value


@pytest.mark.asyncio
async def test_update_message_auto_increments_regeneration_count(initialized_tracker):
    from quackmem.backend import get_backend

    backend = get_backend()
    session = _make_session()
    message = _make_message(session, content="original", regeneration_count=2)

    await backend.create_session(session)
    await backend.insert_message(message)
    await backend.update_message(message.id, "updated")

    rows = await backend.get_messages(session.id)
    assert rows[0]["content"] == "updated"
    assert rows[0]["regeneration_count"] == 3
    assert rows[0]["updated_at"] is not None


@pytest.mark.asyncio
async def test_reserve_assistant_message_inserts_pending_row(initialized_tracker):
    from quackmem.backend import get_backend
    from quackmem.schema.models import MessageReservation

    backend = get_backend()
    session = _make_session()
    await backend.create_session(session)

    reservation = MessageReservation(
        session_id=session.id,
        conversation_id=session.conversation_id,
    )
    await backend.reserve_assistant_message(reservation)

    rows = await backend.get_messages(session.id)
    assert len(rows) == 1
    assert rows[0]["status"] == MessageStatus.pending.value
    assert rows[0]["content"] == ""
    assert str(rows[0]["id"]) == str(reservation.id)


@pytest.mark.asyncio
async def test_finalize_message_completes_reservation(initialized_tracker):
    from quackmem.backend import get_backend
    from quackmem.schema.models import MessageReservation, MessageFinalization

    backend = get_backend()
    session = _make_session()
    await backend.create_session(session)

    reservation = MessageReservation(
        session_id=session.id,
        conversation_id=session.conversation_id,
    )
    await backend.reserve_assistant_message(reservation)

    await backend.finalize_message(MessageFinalization(
        message_id=reservation.id,
        content="final answer",
        token_count=42,
        status=MessageStatus.completed,
    ))

    rows = await backend.get_messages(session.id)
    assert rows[0]["content"] == "final answer"
    assert rows[0]["status"] == MessageStatus.completed.value
    assert rows[0]["token_count"] == 42
    assert rows[0]["updated_at"] is not None


@pytest.mark.asyncio
async def test_finalize_message_records_failure_in_metadata(initialized_tracker):
    from quackmem.backend import get_backend
    from quackmem.schema.models import MessageReservation, MessageFinalization

    backend = get_backend()
    session = _make_session()
    await backend.create_session(session)

    reservation = MessageReservation(
        session_id=session.id,
        conversation_id=session.conversation_id,
        metadata={"trace_id": "abc"},
    )
    await backend.reserve_assistant_message(reservation)

    await backend.finalize_message(MessageFinalization(
        message_id=reservation.id,
        content="",
        status=MessageStatus.failed,
        error="LLM timeout",
    ))

    rows = await backend.get_messages(session.id)
    assert rows[0]["status"] == MessageStatus.failed.value
    assert rows[0]["metadata"]["error"] == "LLM timeout"
    assert rows[0]["metadata"]["trace_id"] == "abc"


@pytest.mark.asyncio
async def test_reap_orphans_marks_old_pending_as_failed(initialized_tracker):
    from datetime import timedelta
    from quackmem.backend import get_backend
    from quackmem.schema.models import MessageReservation

    backend = get_backend()
    session = _make_session()
    await backend.create_session(session)

    # Reserve two assistant messages — both start as pending.
    old_reservation = MessageReservation(
        session_id=session.id,
        conversation_id=session.conversation_id,
        metadata={"trace_id": "old"},
    )
    fresh_reservation = MessageReservation(
        session_id=session.id,
        conversation_id=session.conversation_id,
    )
    await backend.reserve_assistant_message(old_reservation)
    await backend.reserve_assistant_message(fresh_reservation)

    # Backdate the first row so it falls outside the cutoff.
    from sqlalchemy import update as sa_update
    from quackmem.db import tables as _tables
    from quackmem.db import get_session
    from datetime import datetime, UTC
    async with get_session() as db:
        await db.execute(
            sa_update(_tables.tracked_messages)
            .where(_tables.tracked_messages.c.id == old_reservation.id)
            .values(created_at=datetime.now(UTC) - timedelta(hours=1))
        )
        await db.commit()

    reaped = await backend.reap_orphans(older_than=timedelta(minutes=10))

    assert reaped == 1
    rows = {str(r["id"]): r for r in await backend.get_messages(session.id)}
    assert rows[str(old_reservation.id)]["status"] == MessageStatus.failed.value
    assert rows[str(old_reservation.id)]["metadata"]["error"] == "orphaned"
    # Pre-existing metadata is preserved through the merge.
    assert rows[str(old_reservation.id)]["metadata"]["trace_id"] == "old"
    # Fresh row stays pending.
    assert rows[str(fresh_reservation.id)]["status"] == MessageStatus.pending.value


@pytest.mark.asyncio
async def test_reap_orphans_skips_completed_and_failed(initialized_tracker):
    from datetime import timedelta
    from quackmem.backend import get_backend
    from quackmem.schema.models import (
        MessageReservation, MessageFinalization,
    )

    backend = get_backend()
    session = _make_session()
    await backend.create_session(session)

    completed = MessageReservation(
        session_id=session.id, conversation_id=session.conversation_id,
    )
    await backend.reserve_assistant_message(completed)
    await backend.finalize_message(MessageFinalization(
        message_id=completed.id, content="done", status=MessageStatus.completed,
    ))

    # Backdate it so it's older than the cutoff.
    from sqlalchemy import update as sa_update
    from quackmem.db import tables as _tables
    from quackmem.db import get_session
    from datetime import datetime, UTC
    async with get_session() as db:
        await db.execute(
            sa_update(_tables.tracked_messages)
            .where(_tables.tracked_messages.c.id == completed.id)
            .values(created_at=datetime.now(UTC) - timedelta(hours=1))
        )
        await db.commit()

    reaped = await backend.reap_orphans(older_than=timedelta(minutes=1))
    assert reaped == 0


@pytest.mark.asyncio
async def test_get_messages_returns_in_created_at_order(initialized_tracker):
    from quackmem.backend import get_backend

    backend = get_backend()
    session = _make_session()
    msg1 = _make_message(session, content="first")
    msg2 = _make_message(session, content="second")

    await backend.create_session(session)
    await backend.insert_message(msg1)
    await asyncio.sleep(0.01)
    await backend.insert_message(msg2)

    rows = await backend.get_messages(session.id)
    assert [r["content"] for r in rows] == ["first", "second"]
