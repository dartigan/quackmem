"""Tests for the structured-logging helper and the structured fields emitted
by quackmem at runtime."""
from __future__ import annotations

import json
import logging
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from quackmem.core.decorator import track
from quackmem.core.logging import JsonFormatter, configure_logging
from quackmem.wrappers.generic import GenericWrapper


@pytest.fixture(autouse=True)
def _reset_quackmem_logger():
    """Strip handlers between tests so configure_logging is deterministic."""
    log = logging.getLogger("quackmem")
    saved_handlers = list(log.handlers)
    saved_level = log.level
    saved_propagate = log.propagate
    log.handlers.clear()
    yield
    log.handlers = saved_handlers
    log.setLevel(saved_level)
    log.propagate = saved_propagate


class TestJsonFormatter:
    def test_emits_envelope_fields(self):
        record = logging.LogRecord(
            name="quackmem.core.decorator",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        out = json.loads(JsonFormatter().format(record))
        assert out["level"] == "INFO"
        assert out["logger"] == "quackmem.core.decorator"
        assert out["message"] == "hello world"
        assert "timestamp" in out

    def test_extra_fields_passed_through(self):
        record = logging.LogRecord(
            name="quackmem", level=logging.WARNING,
            pathname=__file__, lineno=1, msg="m", args=(), exc_info=None,
        )
        record.session_id = "abc"
        record.error_type = "OperationalError"
        out = json.loads(JsonFormatter().format(record))
        assert out["session_id"] == "abc"
        assert out["error_type"] == "OperationalError"

    def test_non_serialisable_value_falls_back_to_str(self):
        class Weird:
            def __str__(self) -> str:
                return "weird-instance"

        record = logging.LogRecord(
            name="quackmem", level=logging.INFO,
            pathname=__file__, lineno=1, msg="m", args=(), exc_info=None,
        )
        record.obj = Weird()
        out = json.loads(JsonFormatter().format(record))
        assert out["obj"] == "weird-instance"

    def test_exception_field_populated_when_exc_info_set(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="quackmem", level=logging.ERROR,
            pathname=__file__, lineno=1, msg="oops", args=(),
            exc_info=exc_info,
        )
        out = json.loads(JsonFormatter().format(record))
        assert "ValueError: boom" in out["exception"]


class TestConfigureLogging:
    def test_attaches_handler(self):
        configure_logging()
        log = logging.getLogger("quackmem")
        assert any(getattr(h, "_quackmem_managed", False) for h in log.handlers)

    def test_idempotent(self):
        configure_logging()
        configure_logging()
        log = logging.getLogger("quackmem")
        managed = [h for h in log.handlers if getattr(h, "_quackmem_managed", False)]
        # configure_logging removes its prior handler before adding a new one,
        # so the count must stay at 1.
        assert len(managed) == 1

    def test_json_mode_uses_json_formatter(self):
        configure_logging(json=True)
        log = logging.getLogger("quackmem")
        managed = [h for h in log.handlers if getattr(h, "_quackmem_managed", False)]
        assert isinstance(managed[0].formatter, JsonFormatter)

    def test_level_applied(self):
        configure_logging(level="WARNING")
        assert logging.getLogger("quackmem").level == logging.WARNING


class TestStructuredFieldsAtCallSites:
    """Verify decorator error paths attach structured `extra` fields."""

    @pytest.mark.asyncio
    async def test_db_failure_logs_session_and_error_type(self, caplog):
        wrapper = GenericWrapper()
        backend = MagicMock()
        backend.create_session = AsyncMock(
            side_effect=OperationalError("stmt", {}, Exception("db down"))
        )
        backend.insert_message = AsyncMock(return_value=None)
        backend.get_messages = AsyncMock(return_value=[])

        with patch("quackmem.backend.get_backend", return_value=backend), \
             caplog.at_level(logging.ERROR, logger="quackmem.core.decorator"):
            @track(wrapper)
            async def my_fn(messages):
                return "ok"

            await my_fn([{"role": "user", "content": "hi"}])

        records = [r for r in caplog.records if r.message == "Tracking pre-write failed"]
        assert records, "expected error log not found"
        rec = records[0]
        assert hasattr(rec, "session_id")
        assert hasattr(rec, "conversation_id")
        assert rec.error_type == "OperationalError"
        assert rec.path == "async"

    @pytest.mark.asyncio
    async def test_json_log_for_db_failure_is_well_formed(self):
        """End-to-end: configure JSON logging, trigger an error, parse the line."""
        stream = StringIO()
        log = logging.getLogger("quackmem")
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        handler._quackmem_managed = True  # type: ignore[attr-defined]
        log.addHandler(handler)
        log.setLevel(logging.ERROR)
        log.propagate = False

        wrapper = GenericWrapper()
        backend = MagicMock()
        backend.create_session = AsyncMock(
            side_effect=OperationalError("stmt", {}, Exception("down"))
        )
        backend.insert_message = AsyncMock(return_value=None)
        backend.get_messages = AsyncMock(return_value=[])

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_fn(messages):
                return "ok"

            await my_fn([{"role": "user", "content": "hi"}])

        lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
        assert lines, "no log lines emitted"
        payload = json.loads(lines[0])
        assert payload["level"] == "ERROR"
        assert payload["error_type"] == "OperationalError"
        assert payload["path"] == "async"
        assert "session_id" in payload
        assert "conversation_id" in payload
