"""Stress and correctness tests for the fire-and-forget drain mechanism.

These tests verify that:
- The sync decorator path actually registers tasks under load.
- ``wait_pending_writes`` drains every queued write before returning.
- ``shutdown_tracker`` chains drain + dispose without losing messages.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
import sqlalchemy as sa

from quackmem.core.decorator import track
from quackmem.core.tasks import pending_count
from quackmem.wrappers.generic import GenericWrapper


@pytest.mark.asyncio
async def test_many_sync_calls_all_register_and_drain(initialized_tracker):
    """Spawn N sync-decorated calls inside a running loop and confirm every
    one is queued, then fully drained by wait_pending_writes."""
    from quackmem import wait_pending_writes

    wrapper = GenericWrapper()
    sid = uuid.uuid4()

    @track(wrapper, session_id=sid)
    def chat(messages, idx):
        return f"reply-{idx}"

    n_calls = 25
    starting_pending = pending_count()

    # Fire each call from inside the running loop; each schedules a task.
    for i in range(n_calls):
        chat([{"role": "user", "content": f"q{i}"}], idx=i)

    # All N tasks should now be registered.
    assert pending_count() - starting_pending == n_calls, (
        f"expected {n_calls} pending tasks, got {pending_count() - starting_pending}"
    )

    # Drain.
    unfinished = await wait_pending_writes(timeout=10)
    assert unfinished == 0
    assert pending_count() == 0

    # Verify every response made it to Postgres.
    from quackmem.db import get_session, tracked_messages

    async with get_session() as db:
        rows = (
            await db.execute(
                sa.select(tracked_messages)
                .where(tracked_messages.c.session_id == sid)
                .where(tracked_messages.c.role == "assistant")
            )
        ).all()
    contents = sorted(r._mapping["content"] for r in rows)
    assert contents == sorted([f"reply-{i}" for i in range(n_calls)])


@pytest.mark.asyncio
async def test_wait_pending_writes_timeout_returns_count(initialized_tracker):
    """When backend writes are slow, wait_pending_writes(timeout) returns the
    count of still-running tasks rather than hanging forever."""
    from quackmem import wait_pending_writes
    from quackmem.core.tasks import register_pending_task

    async def slow():
        await asyncio.sleep(2)

    # Register a task that we know won't finish inside the timeout.
    task = asyncio.create_task(slow())
    register_pending_task(task)
    assert pending_count() >= 1

    unfinished = await wait_pending_writes(timeout=0.05)
    assert unfinished >= 1

    # Cleanup so the long task doesn't leak into the next test.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_shutdown_tracker_drains_and_disposes(db_url):
    """End-to-end: init, fire writes, shutdown_tracker, verify writes landed
    AND the engine is no longer usable (dispose actually ran)."""
    from quackmem import (
        TrackerConfig,
        init_tracker,
        shutdown_tracker,
        upgrade_db,
        verify_tracker,
    )
    from quackmem.core.exceptions import TrackerConfigError
    from quackmem.db import dispose_engine, get_session, tracked_messages

    # Make sure no leftover engine is around.
    await dispose_engine()
    await asyncio.to_thread(upgrade_db, database_url=db_url)
    init_tracker(TrackerConfig(database_url=db_url))
    await verify_tracker()

    wrapper = GenericWrapper()
    sid = uuid.uuid4()

    @track(wrapper, session_id=sid)
    def chat(messages):
        return "shutdown answer"

    for _ in range(5):
        chat([{"role": "user", "content": "hi"}])

    pending = await shutdown_tracker(drain_timeout=10)
    assert pending == 0

    # After dispose the engine is gone — get_session must now raise.
    with pytest.raises(TrackerConfigError):
        async with get_session() as _db:
            pass

    # Re-init briefly to verify the writes actually landed.
    init_tracker(TrackerConfig(database_url=db_url))
    async with get_session() as db:
        rows = (
            await db.execute(
                sa.select(tracked_messages).where(tracked_messages.c.session_id == sid)
            )
        ).all()
    # All 5 sync calls fire concurrently against a fresh session, so none of
    # them can dedup against the others (no inserts are committed yet when
    # the others read). Each call writes its user message + its assistant
    # response: 5 user + 5 assistant = 10 rows. The thing we actually care
    # about is that all 5 assistant responses landed.
    assistants = [r for r in rows if r._mapping["role"] == "assistant"]
    assert len(assistants) == 5
    assert all(r._mapping["content"] == "shutdown answer" for r in assistants)
    await dispose_engine()
