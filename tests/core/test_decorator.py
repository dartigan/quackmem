"""Tests for the track() decorator."""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quackmem.core.config import TrackerConfig
from quackmem.core.context import get_tracking_context
from quackmem.core.decorator import track
from quackmem.core.exceptions import MetadataValidationError
from quackmem.wrappers.generic import GenericWrapper
from quackmem.schema.enums import MessageRole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> TrackerConfig:
    return TrackerConfig(
        database_url="postgresql+asyncpg://test:test@localhost/test",
    )


def _reset_registry(monkeypatch):
    import quackmem.core.registry as reg_module
    monkeypatch.setattr(reg_module, "_metadata_registry", {})


def _mock_backend():
    """Return a mock backend with async methods."""
    backend = MagicMock()
    backend.create_session = AsyncMock(return_value=None)
    backend.insert_message = AsyncMock(return_value=None)
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrackDecorator:
    """Tests for the track() decorator with mocked DB."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Reset registry before each test."""
        _reset_registry(monkeypatch)

    def test_sync_function_is_called_and_returns_result(self):
        wrapper = GenericWrapper()
        backend = _mock_backend()
        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            def my_func(messages):
                return "the answer"

            result = my_func(["hello"])
        assert result == "the answer"

    @pytest.mark.asyncio
    async def test_async_function_is_called_and_returns_result(self):
        wrapper = GenericWrapper()
        backend = _mock_backend()
        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_async_func(messages):
                return "async result"

            result = await my_async_func(["hi"])
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_auto_generated_session_and_conversation_id(self):
        """When no IDs provided, both UUIDs are auto-generated."""
        wrapper = GenericWrapper()
        captured = {}
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
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
    async def test_session_id_always_auto_generated(self):
        """session_id is always auto-generated, never from kwargs."""
        wrapper = GenericWrapper()
        captured_ids = []
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            # Even if a developer tries to pass session_id as metadata,
            # it gets treated as metadata and is not used as the actual session_id
            @track(wrapper, session_id="should-be-ignored")
            async def my_func(messages):
                captured_ids.append(get_tracking_context().session_id)
                return "ok"

            await my_func(["hello"])

        # Verify session_id is a UUID (auto-generated), not a string
        assert len(captured_ids) == 1
        assert isinstance(captured_ids[0], uuid.UUID)

    @pytest.mark.asyncio
    async def test_provided_conversation_id_is_used(self):
        """When conversation_id is provided, it is used; session_id is always auto-generated."""
        wrapper = GenericWrapper()
        fixed_conv = uuid.uuid4()
        captured = {}
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, conversation_id=fixed_conv)
            async def my_func(messages):
                captured["ctx"] = get_tracking_context()
                return "ok"

            await my_func(["hello"])

        ctx = captured["ctx"]
        # session_id is always auto-generated
        assert isinstance(ctx.session_id, uuid.UUID)
        # conversation_id is the provided one
        assert ctx.conversation_id == fixed_conv

    @pytest.mark.asyncio
    async def test_string_conversation_id_converted_to_uuid(self):
        """String conversation_id is converted to UUID."""
        wrapper = GenericWrapper()
        fixed_conv = uuid.uuid4()
        captured = {}
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, conversation_id=str(fixed_conv))
            async def my_func(messages):
                captured["ctx"] = get_tracking_context()
                return "ok"

            await my_func(["x"])

        assert captured["ctx"].conversation_id == fixed_conv

    @pytest.mark.asyncio
    async def test_metadata_validation_error_propagates(self, monkeypatch):
        """MetadataValidationError is a developer error — it must bubble up."""
        from quackmem.core.registry import register_metadata
        register_metadata({"user_id": str})

        wrapper = GenericWrapper()
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, bad_key="oops")
            async def my_func(messages):
                return "ok"

            with pytest.raises(MetadataValidationError, match="bad_key"):
                await my_func(["hello"])

    @pytest.mark.asyncio
    async def test_tracking_failure_does_not_raise_to_caller(self):
        """If DB write fails, the result is still returned (tracking is fire-and-forget)."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        backend.create_session = AsyncMock(side_effect=RuntimeError("db down"))

        with patch("quackmem.backend.get_backend", return_value=backend):
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
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_func(messages):
                ctx_during["inside"] = get_tracking_context()
                return "ok"

            assert get_tracking_context() is None
            await my_func(["hi"])

        assert ctx_during["inside"] is not None
        assert get_tracking_context() is None  # reset after

    @pytest.mark.asyncio
    async def test_backend_upsert_and_insert_called(self):
        """_fire_write must call create_session then insert_message on the backend."""
        wrapper = GenericWrapper()
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_func(messages):
                return "response"

            await my_func(["input"])

        backend.create_session.assert_called_once()
        backend.insert_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_passed_to_session_model(self):
        """Metadata kwargs (non-reserved) are stored in the session model."""
        wrapper = GenericWrapper()
        captured_args = {}

        async def fake_upsert(session_model):
            captured_args["session"] = session_model

        backend = _mock_backend()
        backend.create_session = fake_upsert

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, user_id="alice")
            async def my_func(messages):
                return "ok"

            await my_func(["hi"])

        # session_id and conversation_id are reserved; user_id becomes metadata
        assert captured_args["session"].metadata.get("user_id") == "alice"
