"""Tests for PostgresBackend.

Unit tests use mock_backend.
Integration tests (marked skip_if_no_db) require CONVO_TRACKER_TEST_DB_URL.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import os
import pytest

from convo_tracker.schema.models import TrackedSession, TrackedMessage

skip_if_no_db = pytest.mark.skipif(
    not os.environ.get("CONVO_TRACKER_TEST_DB_URL"),
    reason="CONVO_TRACKER_TEST_DB_URL not set",
)
from convo_tracker.schema.enums import MessageRole, MessageStatus


# ---------------------------------------------------------------------------
# Unit tests — verify get_backend() singleton and public API shape
# ---------------------------------------------------------------------------

class TestBackendUnit:
    def test_get_backend_returns_postgres_backend(self):
        from convo_tracker.backend import get_backend, PostgresBackend
        b = get_backend()
        assert isinstance(b, PostgresBackend)

    def test_get_backend_is_singleton(self):
        from convo_tracker.backend import get_backend
        b1 = get_backend()
        b2 = get_backend()
        assert b1 is b2

    def test_backend_has_expected_methods(self):
        from convo_tracker.backend import get_backend
        b = get_backend()
        assert callable(b.upsert_session)
        assert callable(b.insert_message)
        assert callable(b.update_status)
        assert callable(b.get_messages)


# ---------------------------------------------------------------------------
# Integration tests — require real Postgres
# ---------------------------------------------------------------------------

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


@skip_if_no_db
class TestBackendIntegration:
    """These tests require CONVO_TRACKER_TEST_DB_URL and will run migrations."""

    @pytest.fixture(autouse=True, scope="class")
    def init_db(self):
        import os
        from convo_tracker import TrackerConfig, init_tracker, upgrade_db
        db_url = os.environ["CONVO_TRACKER_TEST_DB_URL"]
        upgrade_db()
        cfg = TrackerConfig(database_url=db_url, sync_mode=True)
        init_tracker(cfg)

    def test_upsert_session_idempotent(self):
        from convo_tracker.backend import get_backend
        backend = get_backend()
        session = _make_session()

        async def run():
            await backend.upsert_session(session)
            # Second call with same ID must not raise
            await backend.upsert_session(session)

        asyncio.run(run())

    def test_insert_message_writes_row(self):
        from convo_tracker.backend import get_backend
        backend = get_backend()
        session = _make_session()
        message = _make_message(session)

        async def run():
            await backend.upsert_session(session)
            await backend.insert_message(message)
            rows = await backend.get_messages(session.id)
            return rows

        rows = asyncio.run(run())
        assert len(rows) == 1
        assert str(rows[0]["id"]) == str(message.id)

    def test_update_status_only_changes_status(self):
        from convo_tracker.backend import get_backend
        backend = get_backend()
        session = _make_session()
        message = _make_message(session, status=MessageStatus.pending)

        async def run():
            await backend.upsert_session(session)
            await backend.insert_message(message)
            await backend.update_status(message.id, MessageStatus.completed)
            rows = await backend.get_messages(session.id)
            return rows

        rows = asyncio.run(run())
        assert rows[0]["status"] == MessageStatus.completed.value

    def test_get_messages_returns_in_created_at_order(self):
        """Multiple messages should come back ascending by created_at."""
        from convo_tracker.backend import get_backend
        import time
        backend = get_backend()
        session = _make_session()
        msg1 = _make_message(session, content="first")
        time.sleep(0.01)
        msg2 = _make_message(session, content="second")

        async def run():
            await backend.upsert_session(session)
            await backend.insert_message(msg1)
            await backend.insert_message(msg2)
            return await backend.get_messages(session.id)

        rows = asyncio.run(run())
        assert len(rows) == 2
        assert rows[0]["content"] == "first"
        assert rows[1]["content"] == "second"
