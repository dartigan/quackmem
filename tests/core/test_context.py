"""Tests for context variables: TrackingContext, set/get/reset."""
from __future__ import annotations

import asyncio
from uuid import uuid4

from quackmem.core.context import (
    TrackingContext,
    get_tracking_context,
    set_tracking_context,
    reset_tracking_context,
)


class TestTrackingContext:
    def test_get_returns_none_outside_tracked_call(self):
        # Ensure we start clean — ContextVar default is None
        ctx = get_tracking_context()
        assert ctx is None

    def test_set_and_get(self):
        sid = uuid4()
        cid = uuid4()
        tc = TrackingContext(session_id=sid, conversation_id=cid)
        token = set_tracking_context(tc)
        try:
            result = get_tracking_context()
            assert result is tc
            assert result.session_id == sid
            assert result.conversation_id == cid
        finally:
            reset_tracking_context(token)

    def test_reset_restores_none(self):
        tc = TrackingContext(session_id=uuid4(), conversation_id=uuid4())
        token = set_tracking_context(tc)
        reset_tracking_context(token)
        assert get_tracking_context() is None

    def test_nested_context_restores_outer(self):
        outer = TrackingContext(session_id=uuid4(), conversation_id=uuid4())
        inner = TrackingContext(session_id=uuid4(), conversation_id=uuid4())

        token_outer = set_tracking_context(outer)
        assert get_tracking_context() is outer

        token_inner = set_tracking_context(inner)
        assert get_tracking_context() is inner

        reset_tracking_context(token_inner)
        assert get_tracking_context() is outer

        reset_tracking_context(token_outer)
        assert get_tracking_context() is None

    def test_context_is_per_task(self):
        """Two coroutines running concurrently should have independent contexts."""
        results = {}

        async def task_a():
            ctx = TrackingContext(session_id=uuid4(), conversation_id=uuid4())
            token = set_tracking_context(ctx)
            await asyncio.sleep(0)  # yield to allow task_b to run
            results["a"] = get_tracking_context()
            reset_tracking_context(token)

        async def task_b():
            # task_b sets no context — should still see None
            await asyncio.sleep(0)
            results["b"] = get_tracking_context()

        async def run():
            await asyncio.gather(task_a(), task_b())

        asyncio.run(run())
        assert results["a"] is not None
        assert results["b"] is None

    def test_message_id_and_parent_nullable(self):
        ctx = TrackingContext(session_id=uuid4(), conversation_id=uuid4())
        assert ctx.message_id is None
        assert ctx.parent_message_id is None

    def test_message_id_can_be_set(self):
        mid = uuid4()
        ctx = TrackingContext(session_id=uuid4(), conversation_id=uuid4(), message_id=mid)
        assert ctx.message_id == mid

    def test_get_tracking_context_returns_last_context_after_reset(self):
        """After the active context is reset, get_tracking_context returns the last saved ctx."""
        from quackmem.core.context import save_last_tracking_context
        ctx = TrackingContext(session_id=uuid4(), conversation_id=uuid4())
        token = set_tracking_context(ctx)
        # Simulate what the decorator does: save last, then reset active.
        save_last_tracking_context(ctx)
        reset_tracking_context(token)
        # Active is now None, but last should be returned.
        assert get_tracking_context() is ctx

    def test_get_tracking_context_prefers_active_over_last(self):
        """Active context takes precedence over the last-saved one."""
        from quackmem.core.context import save_last_tracking_context
        last_ctx = TrackingContext(session_id=uuid4(), conversation_id=uuid4())
        active_ctx = TrackingContext(session_id=uuid4(), conversation_id=uuid4())
        save_last_tracking_context(last_ctx)
        token = set_tracking_context(active_ctx)
        try:
            assert get_tracking_context() is active_ctx
        finally:
            reset_tracking_context(token)
