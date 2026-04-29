"""Tests for the track() decorator."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quackmem.core.config import TrackerConfig
from quackmem.core.context import get_tracking_context
from quackmem.core.decorator import track
from quackmem.core.exceptions import MetadataValidationError
from quackmem.wrappers.generic import GenericWrapper


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
    backend.update_message = AsyncMock(return_value=None)
    backend.get_messages = AsyncMock(return_value=[])
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
    async def test_session_id_auto_generated_when_not_provided(self):
        """session_id is auto-generated when not provided."""
        wrapper = GenericWrapper()
        captured_ids = []
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_func(messages):
                captured_ids.append(get_tracking_context().session_id)
                return "ok"

            await my_func(["hello"])

        assert len(captured_ids) == 1
        assert isinstance(captured_ids[0], uuid.UUID)

    @pytest.mark.asyncio
    async def test_session_id_from_kwargs_used_when_provided(self):
        """When session_id is provided, it is used."""
        wrapper = GenericWrapper()
        fixed_session = uuid.uuid4()
        captured = {}
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session)
            async def my_func(messages):
                captured["ctx"] = get_tracking_context()
                return "ok"

            await my_func(["hello"])

        ctx = captured["ctx"]
        assert ctx.session_id == fixed_session
        assert isinstance(ctx.conversation_id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_string_session_id_converted_to_uuid(self):
        """String session_id is converted to UUID."""
        wrapper = GenericWrapper()
        fixed_session = uuid.uuid4()
        captured = {}
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=str(fixed_session))
            async def my_func(messages):
                captured["ctx"] = get_tracking_context()
                return "ok"

            await my_func(["x"])

        assert captured["ctx"].session_id == fixed_session

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
    async def test_context_vars_set_during_call_and_available_after(self):
        """Context is active inside the decorated fn and available afterward via last-ctx."""
        wrapper = GenericWrapper()
        ctx_during = {}
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_func(messages):
                ctx_during["inside"] = get_tracking_context()
                return "ok"

            await my_func(["hi"])

        # Context was active inside the call.
        assert ctx_during["inside"] is not None
        # After the call, get_tracking_context() returns the last-saved context (Bug 2 fix).
        ctx_after = get_tracking_context()
        assert ctx_after is not None
        assert ctx_after.session_id == ctx_during["inside"].session_id

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
    async def test_all_input_messages_inserted_on_fresh_session(self):
        """All extracted input messages are inserted plus the response."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        backend.get_messages = AsyncMock(return_value=[])

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_func(messages):
                return "response"

            await my_func([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ])

        backend.create_session.assert_called_once()
        # 2 input messages + 1 response = 3 insert_message calls
        assert backend.insert_message.call_count == 3

    @pytest.mark.asyncio
    async def test_regenerate_message_id_calls_update_not_insert(self):
        """When regenerate_message_id is provided, update_message is called for response."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        regen_id = uuid.uuid4()

        # Existing user message in DB — should be skipped via dedup.
        existing = [
            {"id": uuid.uuid4(), "role": "user", "content": "hi", "created_at": "2024-01-01"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, regenerate_message_id=regen_id)
            async def my_func(messages):
                return "regenerated response"

            await my_func([{"role": "user", "content": "hi"}])

        backend.create_session.assert_called_once()
        # No input messages inserted on regeneration (deduped), response is updated
        assert backend.insert_message.call_count == 0
        backend.update_message.assert_awaited_once_with(
            regen_id,
            "regenerated response",
            regeneration_count=None,
        )

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


    @pytest.mark.asyncio
    async def test_regenerate_on_existing_session_skips_inputs(self):
        """Regeneration skips inserting input messages and updates response in place."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        fixed_session = uuid.uuid4()
        regen_id = uuid.uuid4()

        # Existing messages in DB — input should be deduped, response updated.
        existing = [
            {"id": uuid.uuid4(), "role": "user", "content": "hi", "created_at": "2024-01-01"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session, regenerate_message_id=regen_id)
            async def my_func(messages):
                return "regenerated"

            await my_func([{"role": "user", "content": "hi"}])

        backend.create_session.assert_called_once()
        # No new input messages on regeneration (deduped), response is updated not inserted
        assert backend.insert_message.call_count == 0
        backend.update_message.assert_awaited_once_with(
            regen_id,
            "regenerated",
            regeneration_count=None,
        )

    @pytest.mark.asyncio
    async def test_reused_session_dedups_existing_messages(self):
        """When session_id is reused, only new input messages are inserted."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        fixed_session = uuid.uuid4()

        existing = [
            {
                "id": uuid.uuid4(),
                "role": "system",
                "content": "sys",
                "created_at": "2024-01-01",
            },
            {
                "id": uuid.uuid4(),
                "role": "user",
                "content": "hi",
                "created_at": "2024-01-02",
            },
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session)
            async def my_func(messages):
                return "response"

            await my_func([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "follow up"},
            ])

        backend.create_session.assert_called_once()
        # 1 new input message + 1 response = 2 insert_message calls
        assert backend.insert_message.call_count == 2

    @pytest.mark.asyncio
    async def test_fresh_session_inserts_all_messages(self):
        """Fresh session (empty get_messages) inserts all input messages + response."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        backend.get_messages = AsyncMock(return_value=[])

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                return "response"

            await my_func([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ])

        assert backend.insert_message.call_count == 3


    @pytest.mark.asyncio
    async def test_fire_write_sets_context_message_id(self):
        """After inserting a new response, _fire_write updates ctx.message_id."""
        from quackmem.core.decorator import _fire_write
        from quackmem.core.context import (
            TrackingContext,
            set_tracking_context,
            reset_tracking_context,
        )

        wrapper = GenericWrapper()
        backend = _mock_backend()
        backend.get_messages = AsyncMock(return_value=[])

        sid = uuid.uuid4()
        cid = uuid.uuid4()
        ctx = TrackingContext(session_id=sid, conversation_id=cid)
        token = set_tracking_context(ctx)

        try:
            with patch("quackmem.backend.get_backend", return_value=backend):
                await _fire_write(
                    wrapper,
                    {},
                    ([{"role": "user", "content": "hi"}],),
                    {},
                    "response",
                    sid,
                    cid,
                    {},
                )

            assert ctx.message_id is not None
            assert isinstance(ctx.message_id, uuid.UUID)
        finally:
            reset_tracking_context(token)

    @pytest.mark.asyncio
    async def test_regenerate_sets_context_message_id(self):
        """Regeneration sets ctx.message_id to the existing message ID for chaining."""
        from quackmem.core.decorator import _fire_write
        from quackmem.core.context import (
            TrackingContext,
            set_tracking_context,
            reset_tracking_context,
        )

        wrapper = GenericWrapper()
        backend = _mock_backend()
        backend.get_messages = AsyncMock(return_value=[])
        regen_id = uuid.uuid4()

        sid = uuid.uuid4()
        cid = uuid.uuid4()
        ctx = TrackingContext(session_id=sid, conversation_id=cid)
        token = set_tracking_context(ctx)

        try:
            with patch("quackmem.backend.get_backend", return_value=backend):
                await _fire_write(
                    wrapper,
                    {"regenerate_message_id": regen_id},
                    ([{"role": "user", "content": "hi"}],),
                    {},
                    "regenerated",
                    sid,
                    cid,
                    {},
                )

            # On regeneration, ctx.message_id is set to the existing message ID
            assert ctx.message_id == regen_id
        finally:
            reset_tracking_context(token)

    @pytest.mark.asyncio
    async def test_get_tracking_context_available_after_decorated_call(self):
        """get_tracking_context() returns IDs after the decorated function returns."""
        wrapper = GenericWrapper()
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_func(messages):
                return "ok"

            await my_func(["hi"])

        # Must be non-None after the call — Bug 2 fix.
        ctx = get_tracking_context()
        assert ctx is not None
        assert isinstance(ctx.session_id, uuid.UUID)
        assert isinstance(ctx.conversation_id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_dedup_uses_set_not_positional_prefix(self):
        """Messages already in DB are skipped regardless of their position in the incoming list."""
        wrapper = GenericWrapper()
        backend = _mock_backend()

        existing = [
            {"id": uuid.uuid4(), "role": "user", "content": "existing", "created_at": "2024-01-01"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_func(messages):
                return "response"

            # "existing" appears at position 1, not position 0 — positional dedup would miss it.
            await my_func([
                {"role": "user", "content": "new message"},
                {"role": "user", "content": "existing"},
            ])

        # 1 new input ("new message") + 1 response = 2 inserts; "existing" is deduped.
        assert backend.insert_message.call_count == 2