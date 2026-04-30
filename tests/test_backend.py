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

from quackmem.schema.models import TrackedSession

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
        assert callable(b.reserve_assistant_message)
        assert callable(b.finalize_message)
        assert callable(b.reap_orphans)
        assert callable(b.read_messages)

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
        """Ensure quackmem.db.tables.tracked_sessions/tracked_messages are non-None mocks."""
        mock_table = MagicMock()
        mock_stmt = MagicMock()
        mock_stmt.values.return_value = mock_stmt
        mock_stmt.on_conflict_do_nothing.return_value = mock_stmt
        mock_stmt.where.return_value = mock_stmt
        mock_table.insert.return_value = mock_stmt
        mock_table.update.return_value = mock_stmt
        mock_table.c = MagicMock()
        with patch("quackmem.db.tables.tracked_sessions", mock_table), \
             patch("quackmem.db.tables.tracked_messages", mock_table):
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

    def test_reserve_assistant_message_calls_insert_and_commit(self, mock_db_session):
        from quackmem.backend import PostgresBackend
        from quackmem.schema.models import MessageReservation

        cm, db = mock_db_session
        backend = PostgresBackend()
        reservation = MessageReservation(
            session_id=uuid4(),
            conversation_id=uuid4(),
        )

        mock_stmt = MagicMock()
        with patch("quackmem.backend.get_session", return_value=cm), \
             patch("quackmem.backend.insert", return_value=mock_stmt):
            result = asyncio.run(backend.reserve_assistant_message(reservation))

        assert result is reservation
        db.execute.assert_awaited_once()
        db.commit.assert_awaited_once()

    def test_finalize_message_calls_execute_and_commit(self, mock_db_session):
        from quackmem.backend import PostgresBackend
        from quackmem.schema.models import MessageFinalization
        from quackmem.schema.enums import MessageStatus

        cm, db = mock_db_session
        backend = PostgresBackend()
        finalization = MessageFinalization(
            message_id=uuid4(),
            content="final",
            token_count=12,
            status=MessageStatus.completed,
        )

        mock_stmt = MagicMock()
        mock_stmt.where.return_value = mock_stmt
        mock_stmt.values.return_value = mock_stmt
        with patch("quackmem.backend.get_session", return_value=cm), \
             patch("quackmem.backend.update", return_value=mock_stmt):
            asyncio.run(backend.finalize_message(finalization))

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


# Note: end-to-end backend coverage now lives in tests/integration/, which
# uses real Postgres via pytest-postgresql or QUACKMEM_TEST_DB_URL.
