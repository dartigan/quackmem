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
    backend.reserve_assistant_message = AsyncMock(side_effect=lambda r: r)
    backend.finalize_message = AsyncMock(return_value=None)
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
            @track(wrapper, session_id=uuid.uuid4())
            def my_func(messages):
                return "the answer"

            result = my_func(["hello"])
        assert result == "the answer"

    @pytest.mark.asyncio
    async def test_async_function_is_called_and_returns_result(self):
        wrapper = GenericWrapper()
        backend = _mock_backend()
        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_async_func(messages):
                return "async result"

            result = await my_async_func(["hi"])
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_missing_session_id_raises_value_error(self):
        """session_id is mandatory — auto-generation would silently fragment memory."""
        wrapper = GenericWrapper()
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper)
            async def my_func(messages):
                return "ok"

            with pytest.raises(ValueError, match="session_id is required"):
                await my_func(["hello"])

    @pytest.mark.asyncio
    async def test_conversation_id_auto_generated_when_not_provided(self):
        """conversation_id is optional and auto-generates when not passed."""
        wrapper = GenericWrapper()
        captured = {}
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                captured["ctx"] = get_tracking_context()
                return "ok"

            await my_func(["hello"])

        ctx = captured["ctx"]
        assert ctx is not None
        assert isinstance(ctx.session_id, uuid.UUID)
        assert isinstance(ctx.conversation_id, uuid.UUID)

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
            @track(wrapper, session_id=uuid.uuid4(), conversation_id=fixed_conv)
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
            @track(wrapper, session_id=uuid.uuid4(), conversation_id=str(fixed_conv))
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
            @track(wrapper, session_id=uuid.uuid4(), bad_key="oops")
            async def my_func(messages):
                return "ok"

            with pytest.raises(MetadataValidationError, match="bad_key"):
                await my_func(["hello"])

    @pytest.mark.asyncio
    async def test_db_failure_does_not_raise_to_caller(self):
        """If a DB write fails with a SQLAlchemyError, the result is still returned."""
        from sqlalchemy.exc import OperationalError

        wrapper = GenericWrapper()
        backend = _mock_backend()
        backend.create_session = AsyncMock(
            side_effect=OperationalError("stmt", {}, Exception("db down"))
        )

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                return "still works"

            result = await my_func(["hello"])

        assert result == "still works"

    @pytest.mark.asyncio
    async def test_unexpected_error_in_tracking_propagates(self):
        """Programming errors (TypeError, AttributeError, etc.) inside the
        tracking write must NOT be swallowed — they indicate a bug in
        quackmem or a wrapper, and the caller should see them."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        backend.create_session = AsyncMock(side_effect=TypeError("bug in wrapper"))

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                return "ok"

            with pytest.raises(TypeError, match="bug in wrapper"):
                await my_func(["hello"])

    @pytest.mark.asyncio
    async def test_network_error_in_tracking_does_not_raise(self):
        """OSError (network/connection failures) is treated as expected — caller still gets result."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        backend.create_session = AsyncMock(side_effect=OSError("connection refused"))

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                return "still works"

            result = await my_func(["hi"])

        assert result == "still works"

    @pytest.mark.asyncio
    async def test_context_vars_set_during_call_and_available_after(self):
        """Context is active inside the decorated fn and available afterward via last-ctx."""
        wrapper = GenericWrapper()
        ctx_during = {}
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
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
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                return "response"

            await my_func(["input"])

        backend.create_session.assert_called_once()
        # Bare string input is not extracted by GenericWrapper; assistant uses reserve+finalize.
        assert backend.insert_message.call_count == 0
        backend.reserve_assistant_message.assert_called_once()
        backend.finalize_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_input_messages_inserted_on_fresh_session(self):
        """All extracted input messages are inserted; assistant is reserved+finalized."""
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

        backend.create_session.assert_called_once()
        # 2 input messages inserted; assistant reserved+finalized (not insert_message)
        assert backend.insert_message.call_count == 2
        backend.reserve_assistant_message.assert_called_once()
        backend.finalize_message.assert_called_once()

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
            @track(wrapper, session_id=uuid.uuid4(), regenerate_message_id=regen_id)
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
            @track(wrapper, session_id=uuid.uuid4(), user_id="alice")
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
        # 1 new input message; assistant reserved+finalized.
        assert backend.insert_message.call_count == 1
        backend.reserve_assistant_message.assert_called_once()

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

        assert backend.insert_message.call_count == 2


    @pytest.mark.asyncio
    async def test_pre_write_sets_context_message_id(self):
        """_pre_write reserves an assistant message and updates ctx.message_id."""
        from quackmem.core.decorator import _pre_write
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
                reserved = await _pre_write(
                    wrapper,
                    {},
                    ([{"role": "user", "content": "hi"}],),
                    {},
                    sid,
                    cid,
                    {},
                )

            assert reserved is not None
            assert ctx.message_id == reserved
            assert isinstance(ctx.message_id, uuid.UUID)
        finally:
            reset_tracking_context(token)

    @pytest.mark.asyncio
    async def test_regenerate_sets_context_message_id(self):
        """Regeneration: _pre_write sets ctx.message_id to the existing message ID."""
        from quackmem.core.decorator import _pre_write
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
                reserved = await _pre_write(
                    wrapper,
                    {"regenerate_message_id": regen_id},
                    ([{"role": "user", "content": "hi"}],),
                    {},
                    sid,
                    cid,
                    {},
                )

            # Regeneration returns None (no new reservation) and sets ctx to regen_id.
            assert reserved is None
            assert ctx.message_id == regen_id
        finally:
            reset_tracking_context(token)

    @pytest.mark.asyncio
    async def test_get_tracking_context_available_after_decorated_call(self):
        """get_tracking_context() returns IDs after the decorated function returns."""
        wrapper = GenericWrapper()
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                return "ok"

            await my_func(["hi"])

        # Must be non-None after the call — Bug 2 fix.
        ctx = get_tracking_context()
        assert ctx is not None
        assert isinstance(ctx.session_id, uuid.UUID)
        assert isinstance(ctx.conversation_id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_full_history_resend_inserts_only_new_messages(self):
        """When callers resend the full conversation history, only new tail messages are inserted."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        fixed_session = uuid.uuid4()

        existing = [
            {"id": uuid.uuid4(), "role": "system", "content": "sys", "created_at": "2024-01-01"},
            {"id": uuid.uuid4(), "role": "user", "content": "hi", "created_at": "2024-01-02"},
            {"id": uuid.uuid4(), "role": "assistant", "content": "hello", "created_at": "2024-01-03"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session)
            async def my_func(messages):
                return "response"

            await my_func([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "follow up"},
            ])

        # 1 new input ("follow up"); assistant via reserve+finalize.
        assert backend.insert_message.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_skips_all_input_messages(self):
        """Exact retry of the same request skips all input messages (prefix match)."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        fixed_session = uuid.uuid4()

        existing = [
            {"id": uuid.uuid4(), "role": "user", "content": "hi", "created_at": "2024-01-01"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session)
            async def my_func(messages):
                return "response"

            await my_func([{"role": "user", "content": "hi"}])

        # No new input messages (full prefix match); assistant via reserve+finalize.
        assert backend.insert_message.call_count == 0
        backend.reserve_assistant_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_messages_in_single_request_skipped(self):
        """Adjacent duplicate messages within the same request batch are deduplicated."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        fixed_session = uuid.uuid4()

        existing = [
            {"id": uuid.uuid4(), "role": "user", "content": "hi", "created_at": "2024-01-01"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session)
            async def my_func(messages):
                return "response"

            await my_func([
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ])

        # Prefix match skips first "hi". Second "hi" is adjacent-duplicate → skipped.
        # "ok" is new = 1 input insert; assistant via reserve+finalize.
        assert backend.insert_message.call_count == 1

    @pytest.mark.asyncio
    async def test_repeated_user_query_in_later_turn_is_inserted(self):
        """A repeated query that is not the immediate previous message is inserted."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        fixed_session = uuid.uuid4()

        existing = [
            {"id": uuid.uuid4(), "role": "user", "content": "hi", "created_at": "2024-01-01"},
            {"id": uuid.uuid4(), "role": "assistant", "content": "hello", "created_at": "2024-01-02"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session)
            async def my_func(messages):
                return "response"

            await my_func([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "hi"},
            ])

        # Prefix_len = 2 (user:"hi", assistant:"hello" match DB).
        # Remaining = [user:"hi"]. It differs from last prefix message (assistant)
        # so it is inserted. 1 new input; assistant via reserve+finalize.
        assert backend.insert_message.call_count == 1

    @pytest.mark.asyncio
    async def test_prefix_mismatch_stops_and_inserts_remaining(self):
        """Prefix matching stops at the first mismatch; remaining messages are inserted."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        fixed_session = uuid.uuid4()

        existing = [
            {"id": uuid.uuid4(), "role": "user", "content": "a", "created_at": "2024-01-01"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session)
            async def my_func(messages):
                return "response"

            await my_func([
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "user", "content": "a"},
            ])

        # "a" matches prefix → skipped. "b" is new → inserted. "a" != "b" → inserted.
        # 2 input inserts; assistant via reserve+finalize.
        assert backend.insert_message.call_count == 2

    @pytest.mark.asyncio
    async def test_dedup_normalizes_whitespace(self):
        """Whitespace normalization is applied before prefix and adjacent comparison."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        fixed_session = uuid.uuid4()

        existing = [
            {"id": uuid.uuid4(), "role": "user", "content": "hello   world", "created_at": "2024-01-01"},
        ]
        backend.get_messages = AsyncMock(return_value=existing)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=fixed_session)
            async def my_func(messages):
                return "response"

            await my_func([
                {"role": "user", "content": "  hello world  "},
            ])

        # Normalized content matches → 0 new inputs; assistant via reserve+finalize.
        assert backend.insert_message.call_count == 0
        backend.reserve_assistant_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_function_error_finalizes_message_as_failed(self):
        """When the wrapped fn raises, the reserved message is finalized with status=failed."""
        wrapper = GenericWrapper()
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                raise RuntimeError("model exploded")

            with pytest.raises(RuntimeError, match="model exploded"):
                await my_func([{"role": "user", "content": "hi"}])

        backend.reserve_assistant_message.assert_called_once()
        backend.finalize_message.assert_called_once()
        finalization = backend.finalize_message.call_args.args[0]
        # use_enum_values dumps the enum to its string value at construction.
        assert finalization.status == "failed"
        assert "RuntimeError" in finalization.error
        assert "model exploded" in finalization.error

    @pytest.mark.asyncio
    async def test_pre_write_runs_before_function_body(self):
        """ctx.message_id must be set inside fn body — proving pre-write ran first."""
        wrapper = GenericWrapper()
        backend = _mock_backend()
        captured = {}

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_func(messages):
                captured["mid"] = get_tracking_context().message_id
                return "ok"

            await my_func([{"role": "user", "content": "hi"}])

        assert captured["mid"] is not None
        assert isinstance(captured["mid"], uuid.UUID)
        # And it equals the reserved id used by finalize_message.
        finalization = backend.finalize_message.call_args.args[0]
        assert finalization.message_id == captured["mid"]

    @pytest.mark.asyncio
    async def test_asyncgen_error_finalizes_message_as_failed(self):
        """Streaming error path also finalizes with status=failed."""
        wrapper = GenericWrapper()
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(wrapper, session_id=uuid.uuid4())
            async def my_stream(messages):
                yield "partial"
                raise RuntimeError("stream broke")

            with pytest.raises(RuntimeError, match="stream broke"):
                async for _ in my_stream([{"role": "user", "content": "hi"}]):
                    pass

        backend.reserve_assistant_message.assert_called_once()
        backend.finalize_message.assert_called_once()
        finalization = backend.finalize_message.call_args.args[0]
        assert finalization.status == "failed"