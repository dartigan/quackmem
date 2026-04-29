"""Tests for the track() decorator."""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from convo_tracker.core.config import TrackerConfig
from convo_tracker.core.context import get_tracking_context
from convo_tracker.core.decorator import track, init_decorator
from convo_tracker.core.exceptions import MetadataValidationError
from convo_tracker.wrappers.generic import GenericWrapper
from convo_tracker.schema.enums import MessageRole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(sync_mode: bool = True) -> TrackerConfig:
    return TrackerConfig(
        database_url="postgresql+asyncpg://test:test@localhost/test",
        sync_mode=sync_mode,
    )


def _reset_registry(monkeypatch):
    import convo_tracker.core.registry as reg_module
    monkeypatch.setattr(reg_module, "_metadata_registry", {})


def _noop_sync_write():
    """Return a coroutine mock for sync_write_message."""
    return AsyncMock(return_value=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrackDecorator:
    """Tests for the track() decorator with sync_mode=True and mocked DB."""

    @pytest.fixture(autouse=True)
    def setup(self, sample_config, monkeypatch):
        """Initialize decorator config and reset registry before each test."""
        _reset_registry(monkeypatch)
        init_decorator(sample_config)

    def test_sync_function_is_called_and_returns_result(self):
        wrapper = GenericWrapper()
        with patch("convo_tracker.worker.tasks.sync_write_message", new=_noop_sync_write()):
            @track(wrapper)
            def my_func(messages):
                return "the answer"

            result = my_func(["hello"])
        assert result == "the answer"

    @pytest.mark.asyncio
    async def test_async_function_is_called_and_returns_result(self):
        wrapper = GenericWrapper()
        write_mock = AsyncMock(return_value=None)
        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track(wrapper)
            async def my_async_func(messages):
                return "async result"

            result = await my_async_func(["hi"])
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_auto_generated_session_and_conversation_id(self):
        """When no IDs provided, UUIDs are auto-generated."""
        wrapper = GenericWrapper()
        captured = {}
        write_mock = AsyncMock(return_value=None)

        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track(wrapper)
            async def my_func(messages):
                captured["ctx"] = get_tracking_context()
                return "ok"

            await my_func(["hello"])

        ctx = captured["ctx"]
        assert ctx is not None
        assert isinstance(ctx.session_id, uuid.UUID)
        assert isinstance(ctx.conversation_id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_provided_session_and_conversation_id_are_used(self):
        wrapper = GenericWrapper()
        fixed_session = uuid.uuid4()
        fixed_conv = uuid.uuid4()
        captured = {}
        write_mock = AsyncMock(return_value=None)

        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track(wrapper, session_id=fixed_session, conversation_id=fixed_conv)
            async def my_func(messages):
                captured["ctx"] = get_tracking_context()
                return "ok"

            await my_func(["hello"])

        ctx = captured["ctx"]
        assert ctx.session_id == fixed_session
        assert ctx.conversation_id == fixed_conv

    @pytest.mark.asyncio
    async def test_string_session_id_converted_to_uuid(self):
        wrapper = GenericWrapper()
        fixed_session = uuid.uuid4()
        captured = {}
        write_mock = AsyncMock(return_value=None)

        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track(wrapper, session_id=str(fixed_session))
            async def my_func(messages):
                captured["ctx"] = get_tracking_context()
                return "ok"

            await my_func(["x"])

        assert captured["ctx"].session_id == fixed_session

    @pytest.mark.asyncio
    async def test_metadata_validation_error_propagates(self, monkeypatch):
        """MetadataValidationError is a developer error — it must bubble up."""
        from convo_tracker.core.registry import register_metadata
        register_metadata({"user_id": str})

        wrapper = GenericWrapper()
        write_mock = AsyncMock(return_value=None)

        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track(wrapper, bad_key="oops")
            async def my_func(messages):
                return "ok"

            with pytest.raises(MetadataValidationError, match="bad_key"):
                await my_func(["hello"])

    @pytest.mark.asyncio
    async def test_tracking_failure_does_not_raise_to_caller(self):
        """If DB write fails, the result is still returned (tracking is fire-and-forget)."""
        wrapper = GenericWrapper()
        failing_write = AsyncMock(side_effect=RuntimeError("db down"))

        with patch("convo_tracker.worker.tasks.sync_write_message", failing_write):
            @track(wrapper)
            async def my_func(messages):
                return "still works"

            result = await my_func(["hello"])

        assert result == "still works"

    @pytest.mark.asyncio
    async def test_context_vars_set_during_call_and_reset_after(self):
        """Context is active inside the decorated fn but None afterward."""
        wrapper = GenericWrapper()
        ctx_during = {}
        write_mock = AsyncMock(return_value=None)

        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track(wrapper)
            async def my_func(messages):
                ctx_during["inside"] = get_tracking_context()
                return "ok"

            assert get_tracking_context() is None
            await my_func(["hi"])

        assert ctx_during["inside"] is not None
        assert get_tracking_context() is None  # reset after

    @pytest.mark.asyncio
    async def test_sync_mode_calls_sync_write_message(self):
        """With sync_mode=True, _fire_write should call sync_write_message."""
        wrapper = GenericWrapper()
        write_mock = AsyncMock(return_value=None)

        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track(wrapper)
            async def my_func(messages):
                return "response"

            await my_func(["input"])

        write_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_passed_to_session_model(self):
        """Metadata kwargs (non-reserved) are stored in the session model."""
        wrapper = GenericWrapper()
        captured_args = {}

        async def capture_write(session_model, message_model):
            captured_args["session"] = session_model
            captured_args["message"] = message_model

        with patch("convo_tracker.worker.tasks.sync_write_message", capture_write):
            @track(wrapper, user_id="alice")
            async def my_func(messages):
                return "ok"

            await my_func(["hi"])

        # session_id and conversation_id are reserved; user_id becomes metadata
        assert captured_args["session"].metadata.get("user_id") == "alice"
