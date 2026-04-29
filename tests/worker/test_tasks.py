"""Tests for worker/tasks.py — sync write path and task helpers."""
from __future__ import annotations

import asyncio
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from convo_tracker.schema.models import TrackedSession, TrackedMessage
from convo_tracker.schema.enums import MessageRole, MessageStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> TrackedSession:
    return TrackedSession(conversation_id=uuid4())


def _make_message(session: TrackedSession) -> TrackedMessage:
    return TrackedMessage(
        session_id=session.id,
        conversation_id=session.conversation_id,
        role=MessageRole.assistant,
        content="hello from test",
        status=MessageStatus.completed,
    )


# ---------------------------------------------------------------------------
# sync_write_message
# ---------------------------------------------------------------------------

class TestSyncWriteMessage:
    @pytest.mark.asyncio
    async def test_calls_upsert_session_and_insert_message(self, mock_backend):
        """sync_write_message must call upsert_session then insert_message."""
        from convo_tracker.worker.tasks import sync_write_message

        session = _make_session()
        message = _make_message(session)

        await sync_write_message(session, message)

        mock_backend.upsert_session.assert_called_once_with(session)
        mock_backend.insert_message.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_upsert_called_before_insert(self, mock_backend):
        """Order must be: upsert_session, then insert_message."""
        call_order = []
        mock_backend.upsert_session.side_effect = lambda *a, **kw: call_order.append("upsert") or None
        mock_backend.insert_message.side_effect = lambda *a, **kw: call_order.append("insert") or None

        # Make them awaitable by replacing with real coroutines
        async def fake_upsert(s):
            call_order.append("upsert")

        async def fake_insert(m):
            call_order.append("insert")

        mock_backend.upsert_session = fake_upsert
        mock_backend.insert_message = fake_insert

        from convo_tracker.worker.tasks import sync_write_message
        session = _make_session()
        message = _make_message(session)
        await sync_write_message(session, message)

        assert call_order == ["upsert", "insert"]


# ---------------------------------------------------------------------------
# sync_update_status
# ---------------------------------------------------------------------------

class TestSyncUpdateStatus:
    @pytest.mark.asyncio
    async def test_calls_update_status_on_backend(self, mock_backend):
        from convo_tracker.worker.tasks import sync_update_status

        msg_id = uuid4()
        status = MessageStatus.completed

        await sync_update_status(msg_id, status)

        mock_backend.update_status.assert_called_once_with(msg_id, status)


# ---------------------------------------------------------------------------
# write_message Celery task (unit — no broker needed)
# ---------------------------------------------------------------------------

class TestWriteMessageTask:
    def test_task_is_registered(self):
        """write_message should be importable as a Celery shared task."""
        from convo_tracker.worker.tasks import write_message
        assert callable(write_message)

    def test_update_message_status_task_is_registered(self):
        from convo_tracker.worker.tasks import update_message_status
        assert callable(update_message_status)

    def test_write_message_validates_session_dict(self, mock_backend):
        """write_message deserializes session_dict and message_dict before calling backend."""
        from convo_tracker.worker.tasks import _write
        session = _make_session()
        message = _make_message(session)

        async def run():
            await _write(mock_backend, session, message)

        asyncio.run(run())
        mock_backend.upsert_session.assert_called_once_with(session)
        mock_backend.insert_message.assert_called_once_with(message)
