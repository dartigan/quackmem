"""End-to-end integration tests for the @track decorator against a real Postgres.

These tests verify the full path: decorator → wrapper → backend → SQLAlchemy →
Postgres → query back. They catch issues that mocked unit tests can't, such as
schema mismatches, asyncpg driver quirks, and concurrency bugs.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
import sqlalchemy as sa

from quackmem.core.decorator import track
from quackmem.wrappers.generic import GenericWrapper


@pytest.mark.asyncio
async def test_async_decorator_writes_session_and_messages(initialized_tracker):
    wrapper = GenericWrapper()
    sid = uuid.uuid4()

    @track(wrapper, session_id=sid, user_id="alice")
    async def chat(messages):
        return "the answer"

    result = await chat([{"role": "user", "content": "hello"}])
    assert result == "the answer"

    # Verify rows persisted.
    from quackmem.db import get_session, tracked_sessions, tracked_messages

    async with get_session() as db:
        sess_rows = (
            await db.execute(sa.select(tracked_sessions).where(tracked_sessions.c.id == sid))
        ).all()
        msg_rows = (
            await db.execute(
                sa.select(tracked_messages)
                .where(tracked_messages.c.session_id == sid)
                .order_by(tracked_messages.c.created_at.asc())
            )
        ).all()

    assert len(sess_rows) == 1
    # 1 input + 1 response
    assert len(msg_rows) == 2
    roles = [r._mapping["role"] for r in msg_rows]
    assert roles == ["user", "assistant"]
    assert msg_rows[1]._mapping["content"] == "the answer"


@pytest.mark.asyncio
async def test_resend_full_history_is_deduplicated(initialized_tracker):
    wrapper = GenericWrapper()
    sid = uuid.uuid4()

    @track(wrapper, session_id=sid)
    async def chat(messages):
        # Echo the last user message so each turn produces a distinct response.
        return f"reply: {messages[-1]['content']}"

    await chat([{"role": "user", "content": "turn 1"}])
    await chat([
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply: turn 1"},
        {"role": "user", "content": "turn 2"},
    ])

    from quackmem.db import get_session, tracked_messages

    async with get_session() as db:
        msg_rows = (
            await db.execute(
                sa.select(tracked_messages)
                .where(tracked_messages.c.session_id == sid)
                .order_by(tracked_messages.c.created_at.asc())
            )
        ).all()

    contents = [r._mapping["content"] for r in msg_rows]
    # First call: user "turn 1", assistant "reply: turn 1".
    # Second call: prefix-dedup skips the first two; inserts user "turn 2"
    # and assistant "reply: turn 2".
    assert contents == ["turn 1", "reply: turn 1", "turn 2", "reply: turn 2"]


@pytest.mark.asyncio
async def test_regenerate_updates_existing_message(initialized_tracker):
    wrapper = GenericWrapper()
    sid = uuid.uuid4()

    @track(wrapper, session_id=sid)
    async def chat(messages):
        return "original answer"

    from quackmem import get_tracking_context

    await chat([{"role": "user", "content": "q"}])
    ctx = get_tracking_context()
    assert ctx is not None and ctx.message_id is not None
    msg_id = ctx.message_id

    @track(wrapper, session_id=sid, regenerate_message_id=msg_id)
    async def regen(messages):
        return "regenerated answer"

    await regen([{"role": "user", "content": "q"}])

    from quackmem.db import get_session, tracked_messages

    async with get_session() as db:
        row = (
            await db.execute(
                sa.select(tracked_messages).where(tracked_messages.c.id == msg_id)
            )
        ).one()
    assert row._mapping["content"] == "regenerated answer"
    assert row._mapping["regeneration_count"] == 1


@pytest.mark.asyncio
async def test_concurrent_writes_to_distinct_sessions(initialized_tracker):
    """Many tracked calls firing in parallel must all succeed without
    deadlocks or races."""
    wrapper = GenericWrapper()

    async def one_call(i: int) -> None:
        @track(wrapper, session_id=uuid.uuid4())
        async def chat(messages):
            await asyncio.sleep(0.01)
            return "ok"

        await chat([{"role": "user", "content": f"m{i}"}])

    await asyncio.gather(*[one_call(i) for i in range(20)])

    from quackmem.db import get_session, tracked_sessions

    async with get_session() as db:
        count = (await db.execute(sa.select(sa.func.count()).select_from(tracked_sessions))).scalar()
    assert count == 20


@pytest.mark.asyncio
async def test_sync_decorator_in_loop_drains_via_wait_pending_writes(initialized_tracker):
    """The sync decorator path schedules a fire-and-forget task. Without
    wait_pending_writes the test would race against the async write."""
    from quackmem import wait_pending_writes

    wrapper = GenericWrapper()
    sid = uuid.uuid4()

    @track(wrapper, session_id=sid)
    def chat(messages):
        return "sync answer"

    chat([{"role": "user", "content": "hi"}])
    pending_left = await wait_pending_writes(timeout=5)
    assert pending_left == 0

    from quackmem.db import get_session, tracked_messages

    async with get_session() as db:
        rows = (
            await db.execute(
                sa.select(tracked_messages).where(tracked_messages.c.session_id == sid)
            )
        ).all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_read_messages_returns_decorator_history(initialized_tracker):
    """End-to-end: decorator writes a turn, ``read_messages`` returns it."""
    from quackmem import read_messages

    sid = uuid.uuid4()
    wrapper = GenericWrapper()

    @track(wrapper, session_id=sid)
    async def chat(messages: list[dict]) -> str:
        return "the answer"

    await chat([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what is 2+2?"},
    ])

    rows = await read_messages(sid)
    contents = [r.content for r in rows]
    assert contents == ["sys", "what is 2+2?", "the answer"]


@pytest.mark.asyncio
async def test_read_messages_uses_default_limit(initialized_tracker):
    """``limit`` omitted → ``TrackerConfig.default_read_limit`` (default 10) applies."""
    from quackmem import read_messages
    from quackmem.backend import get_backend
    from quackmem.schema.enums import MessageRole, MessageStatus
    from quackmem.schema.models import TrackedMessage, TrackedSession

    backend = get_backend()
    sid = uuid.uuid4()
    cid = uuid.uuid4()
    await backend.create_session(TrackedSession(id=sid, conversation_id=cid))

    for i in range(15):
        await backend.insert_message(TrackedMessage(
            session_id=sid,
            conversation_id=cid,
            role=MessageRole.user,
            content=f"m{i}",
            status=MessageStatus.completed,
        ))

    rows = await read_messages(sid)
    assert len(rows) == 10
    assert rows[0].content == "m5"
    assert rows[-1].content == "m14"
