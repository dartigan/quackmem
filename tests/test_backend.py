"""Tests for PostgresBackend.

Unit tests use mock_backend.
Integration tests (marked skip_if_no_db) require QUACKMEM_TEST_DB_URL.
"""
from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from quackmem.schema.enums import MessageRole, MessageStatus
from quackmem.schema.models import TrackedSession, TrackedMessage

skip_if_no_db = pytest.mark.skipif(
    not os.environ.get("QUACKMEM_TEST_DB_URL"),
    reason="QUACKMEM_TEST_DB_URL not set",
)


# ---------------------------------------------------------------------------
# Unit tests — verify get_backend() singleton and public API shape
# ---------------------------------------------------------------------------

class TestBackendUnit:
    def test_get_backend_returns_postgres_backend(self):
        from quackmem.backend import get_backend, PostgresBackend
        b = get_backend()
        assert isinstance(b, PostgresBackend)

    def test_get_backend_is_singleton(self):
        from quackmem.backend import get_backend
        b1 = get_backend()
        b2 = get_backend()
        assert b1 is b2

    def test_backend_has_expected_methods(self):
        from quackmem.backend import get_backend
        b = get_backend()
        assert callable(b.create_session)
        assert callable(b.insert_message)
        assert callable(b.update_message)
        assert callable(b.update_status)
        assert callable(b.get_messages)

    def test_build_tables_is_idempotent(self):
        """Calling build_tables twice must not raise (Bug 5 fix)."""
        from quackmem.db.tables import build_tables
        t1, m1 = build_tables(schema=None, prefix="")
        t2, m2 = build_tables(schema=None, prefix="")
        assert t1 is t2
        assert m1 is m2


# ---------------------------------------------------------------------------
# Unit tests — is_retryable_sqlalchemy_error
# ---------------------------------------------------------------------------

class TestIsRetryableSqlalchemyError:
    def test_returns_true_for_retryable_pgcode(self):
        from quackmem.backend import is_retryable_sqlalchemy_error, RETRYABLE_PG_CODES
        from sqlalchemy.exc import OperationalError

        for pgcode in RETRYABLE_PG_CODES:
            orig = MagicMock()
            orig.pgcode = pgcode
            exc = OperationalError("statement", {}, orig)
            assert is_retryable_sqlalchemy_error(exc) is True, f"Expected True for pgcode={pgcode}"

    def test_returns_false_for_non_retryable_pgcode(self):
        from quackmem.backend import is_retryable_sqlalchemy_error
        from sqlalchemy.exc import OperationalError

        orig = MagicMock()
        orig.pgcode = "42P01"  # undefined_table — not retryable
        exc = OperationalError("statement", {}, orig)
        assert is_retryable_sqlalchemy_error(exc) is False

    def test_returns_false_for_non_operational_error(self):
        from quackmem.backend import is_retryable_sqlalchemy_error

        assert is_retryable_sqlalchemy_error(ValueError("boom")) is False
        assert is_retryable_sqlalchemy_error(RuntimeError("network gone")) is False
        assert is_retryable_sqlalchemy_error(Exception("generic")) is False


# ---------------------------------------------------------------------------
# Unit tests — mock DB calls for create_session and update_message
# ---------------------------------------------------------------------------

class TestBackendMockCalls:
    @pytest.fixture()
    def mock_db_session(self):
        """Return a mock async context manager that mimics get_session()."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=db)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm, db

    @pytest.fixture(autouse=True)
    def patch_tables(self):
        """Ensure tracked_sessions and tracked_messages are non-None mocks."""
        mock_table = MagicMock()
        # Make insert/update/select return something that SQLAlchemy-like code can call .values() on
        mock_stmt = MagicMock()
        mock_stmt.values.return_value = mock_stmt
        mock_stmt.on_conflict_do_nothing.return_value = mock_stmt
        mock_stmt.where.return_value = mock_stmt
        mock_table.insert.return_value = mock_stmt
        mock_table.update.return_value = mock_stmt
        mock_table.c = MagicMock()
        with patch("quackmem.backend.tracked_sessions", mock_table), \
             patch("quackmem.backend.tracked_messages", mock_table):
            yield

    def test_create_session_calls_execute_and_commit(self, mock_db_session):
        from quackmem.backend import PostgresBackend

        cm, db = mock_db_session
        backend = PostgresBackend()
        session = TrackedSession(conversation_id=uuid4())

        mock_stmt = MagicMock()
        mock_stmt.on_conflict_do_nothing.return_value = mock_stmt
        with patch("quackmem.backend.get_session", return_value=cm), \
             patch("quackmem.backend.insert", return_value=mock_stmt):
            asyncio.run(backend.create_session(session))

        db.execute.assert_awaited_once()
        db.commit.assert_awaited_once()

    def test_update_message_calls_execute_and_commit(self, mock_db_session):
        from quackmem.backend import PostgresBackend

        cm, db = mock_db_session
        backend = PostgresBackend()
        message_id = uuid4()

        mock_stmt = MagicMock()
        mock_stmt.where.return_value = mock_stmt
        mock_stmt.values.return_value = mock_stmt
        with patch("quackmem.backend.get_session", return_value=cm), \
             patch("quackmem.backend.update", return_value=mock_stmt):
            asyncio.run(backend.update_message(message_id, "new content", regeneration_count=1))

        db.execute.assert_awaited_once()
        db.commit.assert_awaited_once()

    def test_update_message_without_regeneration_count(self, mock_db_session):
        from quackmem.backend import PostgresBackend

        cm, db = mock_db_session
        backend = PostgresBackend()
        message_id = uuid4()

        mock_stmt = MagicMock()
        mock_stmt.where.return_value = mock_stmt
        mock_stmt.values.return_value = mock_stmt
        with patch("quackmem.backend.get_session", return_value=cm), \
             patch("quackmem.backend.update", return_value=mock_stmt):
            asyncio.run(backend.update_message(message_id, "updated content"))

        db.execute.assert_awaited_once()
        db.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# Helpers shared by integration tests
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


# ---------------------------------------------------------------------------
# Integration tests — require real Postgres
# ---------------------------------------------------------------------------

@skip_if_no_db
class TestBackendIntegration:
    """These tests require QUACKMEM_TEST_DB_URL and will run migrations."""

    @pytest.fixture(autouse=True, scope="class")
    def init_db(self):
        import os
        from quackmem import TrackerConfig, init_tracker, upgrade_db
        db_url = os.environ["QUACKMEM_TEST_DB_URL"]
        upgrade_db()
        cfg = TrackerConfig(database_url=db_url, sync_mode=True)
        init_tracker(cfg)

    def test_create_session_idempotent(self):
        from quackmem.backend import get_backend
        backend = get_backend()
        session = _make_session()

        async def run():
            await backend.create_session(session)
            # Second call with same ID must not raise
            await backend.create_session(session)

        asyncio.run(run())

    def test_insert_message_writes_row(self):
        from quackmem.backend import get_backend
        backend = get_backend()
        session = _make_session()
        message = _make_message(session)

        async def run():
            await backend.create_session(session)
            await backend.insert_message(message)
            rows = await backend.get_messages(session.id)
            return rows

        rows = asyncio.run(run())
        assert len(rows) == 1
        assert str(rows[0]["id"]) == str(message.id)

    def test_update_status_only_changes_status(self):
        from quackmem.backend import get_backend
        backend = get_backend()
        session = _make_session()
        message = _make_message(session, status=MessageStatus.pending)

        async def run():
            await backend.create_session(session)
            await backend.insert_message(message)
            await backend.update_status(message.id, MessageStatus.completed)
            rows = await backend.get_messages(session.id)
            return rows

        rows = asyncio.run(run())
        assert rows[0]["status"] == MessageStatus.completed.value

    def test_update_message_updates_content_and_updated_at(self):
        from quackmem.backend import get_backend
        backend = get_backend()
        session = _make_session()
        message = _make_message(session, content="original")

        async def run():
            await backend.create_session(session)
            await backend.insert_message(message)
            await backend.update_message(message.id, "updated", regeneration_count=1)
            rows = await backend.get_messages(session.id)
            return rows

        rows = asyncio.run(run())
        assert rows[0]["content"] == "updated"
        assert rows[0]["regeneration_count"] == 1
        assert rows[0]["updated_at"] is not None

    def test_update_message_auto_increments_regeneration_count(self):
        from quackmem.backend import get_backend
        backend = get_backend()
        session = _make_session()
        message = _make_message(session, content="original", regeneration_count=2)

        async def run():
            await backend.create_session(session)
            await backend.insert_message(message)
            await backend.update_message(message.id, "updated")
            rows = await backend.get_messages(session.id)
            return rows

        rows = asyncio.run(run())
        assert rows[0]["content"] == "updated"
        assert rows[0]["regeneration_count"] == 3
        assert rows[0]["updated_at"] is not None

    def test_get_messages_returns_in_created_at_order(self):
        """Multiple messages should come back ascending by created_at."""
        from quackmem.backend import get_backend
        import time
        backend = get_backend()
        session = _make_session()
        msg1 = _make_message(session, content="first")
        time.sleep(0.01)
        msg2 = _make_message(session, content="second")

        async def run():
            await backend.create_session(session)
            await backend.insert_message(msg1)
            await backend.insert_message(msg2)
            return await backend.get_messages(session.id)

        rows = asyncio.run(run())
        assert len(rows) == 2
        assert rows[0]["content"] == "first"
        assert rows[1]["content"] == "second"
