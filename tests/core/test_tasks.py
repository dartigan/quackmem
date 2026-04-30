"""Tests for the pending-task registry used by the sync decorator path."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quackmem.core.tasks import (
    pending_count,
    register_pending_task,
    wait_pending_writes,
)
from quackmem.core.decorator import track
from quackmem.wrappers.generic import GenericWrapper


@pytest.fixture(autouse=True)
def _clear_pending():
    """Make sure each test starts with an empty registry."""
    from quackmem.core import tasks
    tasks._pending.clear()
    yield
    tasks._pending.clear()


class TestPendingTaskRegistry:
    @pytest.mark.asyncio
    async def test_empty_registry_returns_zero_immediately(self):
        assert pending_count() == 0
        result = await wait_pending_writes()
        assert result == 0

    @pytest.mark.asyncio
    async def test_registered_task_drains(self):
        completed = []

        async def work():
            await asyncio.sleep(0.01)
            completed.append(True)

        task = asyncio.create_task(work())
        register_pending_task(task)
        assert pending_count() == 1

        result = await wait_pending_writes()
        assert result == 0
        assert completed == [True]
        # Done callback removes finished tasks.
        assert pending_count() == 0

    @pytest.mark.asyncio
    async def test_timeout_returns_count_of_unfinished(self):
        async def slow():
            await asyncio.sleep(1)

        task = asyncio.create_task(slow())
        register_pending_task(task)

        result = await wait_pending_writes(timeout=0.05)
        assert result == 1

        # Cleanup: cancel the long-running task so it doesn't leak.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_multiple_tasks_all_drain(self):
        seen = []

        async def work(n):
            await asyncio.sleep(0.01)
            seen.append(n)

        for i in range(5):
            register_pending_task(asyncio.create_task(work(i)))
        assert pending_count() == 5

        result = await wait_pending_writes(timeout=1)
        assert result == 0
        assert sorted(seen) == [0, 1, 2, 3, 4]


class TestSyncDecoratorRegistersPendingTask:
    """The sync decorator path uses register_pending_task when called from
    inside an event loop (FastAPI, etc.) — verify wait_pending_writes drains
    those writes."""

    @pytest.mark.asyncio
    async def test_sync_call_inside_loop_registers_and_drains(self):
        wrapper = GenericWrapper()
        backend = MagicMock()
        backend.create_session = AsyncMock(return_value=None)
        backend.insert_message = AsyncMock(return_value=None)
        backend.update_message = AsyncMock(return_value=None)
        backend.get_messages = AsyncMock(return_value=[])
        backend.reserve_assistant_message = AsyncMock(side_effect=lambda r: r)
        backend.finalize_message = AsyncMock(return_value=None)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            def my_sync_func(messages):
                return "ok"

            # Calling sync function from inside a running loop schedules a task.
            result = my_sync_func([{"role": "user", "content": "hi"}])
            assert result == "ok"

            # Task is registered and not yet complete.
            assert pending_count() >= 1

            # Drain it.
            unfinished = await wait_pending_writes(timeout=2)
            assert unfinished == 0

        backend.create_session.assert_awaited_once()
        backend.insert_message.assert_awaited()
