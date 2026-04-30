"""Tests for GenericWrapper and track_conversation decorator."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quackmem.wrappers.generic import GenericWrapper, track_conversation
from quackmem.schema.enums import MessageRole


def _mock_backend():
    backend = MagicMock()
    backend.create_session = AsyncMock(return_value=None)
    backend.insert_message = AsyncMock(return_value=None)
    backend.update_message = AsyncMock(return_value=None)
    backend.get_messages = AsyncMock(return_value=[])
    backend.reserve_assistant_message = AsyncMock(side_effect=lambda r: r)
    backend.finalize_message = AsyncMock(return_value=None)
    return backend


# ---------------------------------------------------------------------------
# GenericWrapper.extract_messages
# ---------------------------------------------------------------------------

class TestGenericWrapperExtractMessages:
    def setup_method(self):
        self.wrapper = GenericWrapper()

    def test_string_input_becomes_user_message(self):
        result = self.wrapper.extract_messages(("hello",), {})
        assert len(result) == 1
        assert result[0].role == MessageRole.user
        assert result[0].content == "hello"

    def test_list_of_dicts_with_role_key(self):
        msgs = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        result = self.wrapper.extract_messages((msgs,), {})
        assert len(result) == 2
        assert result[0].role == MessageRole.user
        assert result[1].role == MessageRole.assistant

    def test_messages_kwarg_fallback(self):
        msgs = [{"role": "user", "content": "via kwarg"}]
        result = self.wrapper.extract_messages((), {"messages": msgs})
        assert len(result) == 1

    def test_input_kwarg_fallback(self):
        result = self.wrapper.extract_messages((), {"input": "hi"})
        assert len(result) == 1
        assert result[0].content == "hi"

    def test_dict_without_role_is_skipped(self):
        msgs = [{"content": "no role here"}]
        result = self.wrapper.extract_messages((msgs,), {})
        assert result == []

    def test_empty_list(self):
        result = self.wrapper.extract_messages(([], ), {})
        assert result == []

    def test_invalid_role_skipped(self):
        msgs = [
            {"role": "user", "content": "valid"},
            {"role": "bogus_role", "content": "invalid"},
        ]
        result = self.wrapper.extract_messages((msgs,), {})
        assert len(result) == 1

    def test_tool_calls_extracted(self):
        calls = [{"id": "tc1", "type": "function"}]
        msgs = [{"role": "assistant", "content": "", "tool_calls": calls}]
        result = self.wrapper.extract_messages((msgs,), {})
        assert result[0].tool_calls == calls


# ---------------------------------------------------------------------------
# GenericWrapper.extract_response
# ---------------------------------------------------------------------------

class TestGenericWrapperExtractResponse:
    def setup_method(self):
        self.wrapper = GenericWrapper()

    def test_string_result(self):
        result = self.wrapper.extract_response("final output")
        assert result.role == MessageRole.assistant
        assert result.content == "final output"

    def test_dict_with_content_key(self):
        result = self.wrapper.extract_response({"content": "response text"})
        assert result.role == MessageRole.assistant
        assert result.content == "response text"

    def test_dict_with_output_key_fallback(self):
        result = self.wrapper.extract_response({"output": "from output key"})
        assert result.role == MessageRole.assistant
        assert result.content == "from output key"

    def test_dict_without_content_or_output_stringified(self):
        d = {"key": "val"}
        result = self.wrapper.extract_response(d)
        assert result.role == MessageRole.assistant
        assert isinstance(result.content, str)

    def test_non_string_non_dict_stringified(self):
        result = self.wrapper.extract_response(12345)
        assert result.content == "12345"

    def test_extracts_tool_calls_from_dict_response(self):
        calls = [
            {"id": "c1", "name": "search"},
            {"id": "c2", "name": "fetch"},
        ]
        result = self.wrapper.extract_response({"content": "thinking", "tool_calls": calls})
        assert result.tool_calls == calls

    def test_string_response_has_no_tool_calls(self):
        result = self.wrapper.extract_response("just text")
        assert result.tool_calls is None

    def test_non_list_tool_calls_ignored(self):
        """A malformed ``tool_calls`` (not a list) must not crash; just skip."""
        result = self.wrapper.extract_response({"content": "x", "tool_calls": "oops"})
        assert result.tool_calls is None


# ---------------------------------------------------------------------------
# GenericWrapper.extract_token_count
# ---------------------------------------------------------------------------

class TestGenericWrapperExtractTokenCount:
    def setup_method(self):
        self.wrapper = GenericWrapper()

    def test_extracts_total_tokens(self):
        result = {"usage": {"total_tokens": 77}}
        assert self.wrapper.extract_token_count(result) == 77

    def test_extracts_completion_tokens_fallback(self):
        result = {"usage": {"completion_tokens": 30}}
        assert self.wrapper.extract_token_count(result) == 30

    def test_returns_none_for_string_result(self):
        assert self.wrapper.extract_token_count("text") is None

    def test_returns_none_when_no_usage(self):
        assert self.wrapper.extract_token_count({}) is None


# ---------------------------------------------------------------------------
# track_conversation decorator
# ---------------------------------------------------------------------------

class TestTrackConversationDecorator:
    @pytest.mark.asyncio
    async def test_async_function_wrapped(self):
        backend = _mock_backend()
        with patch("quackmem.backend.get_backend", return_value=backend):
            @track_conversation(session_id=uuid.uuid4())
            async def my_llm_call(messages):
                return "llm output"

            result = await my_llm_call([{"role": "user", "content": "test"}])

        assert result == "llm output"
        backend.create_session.assert_called_once()
        # 1 input message inserted; assistant via reserve+finalize.
        assert backend.insert_message.call_count == 1
        backend.reserve_assistant_message.assert_called_once()
        backend.finalize_message.assert_called_once()

    def test_sync_function_wrapped(self):
        backend = _mock_backend()
        with patch("quackmem.backend.get_backend", return_value=backend):
            @track_conversation(session_id=uuid.uuid4())
            def my_sync_call(messages):
                return "sync output"

            result = my_sync_call([{"role": "user", "content": "hello"}])

        assert result == "sync output"
        backend.create_session.assert_called_once()
        # 1 input message inserted; assistant via reserve+finalize.
        assert backend.insert_message.call_count == 1
        backend.reserve_assistant_message.assert_called_once()
        backend.finalize_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_metadata_passed(self):
        captured = {}
        backend = _mock_backend()

        async def capture_upsert(session):
            captured["session"] = session

        backend.create_session = capture_upsert

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track_conversation(session_id=uuid.uuid4(), user_id="bob")
            async def my_llm_call(messages):
                return "ok"

            await my_llm_call(["hi"])

        assert captured["session"].metadata.get("user_id") == "bob"

    @pytest.mark.asyncio
    async def test_tracking_error_does_not_break_caller(self):
        from sqlalchemy.exc import OperationalError

        backend = _mock_backend()
        # Storage-layer failures are caught and logged; the caller still gets
        # its result. (Programming bugs are intentionally re-raised — see
        # tests/core/test_decorator.py::test_unexpected_error_in_tracking_propagates.)
        backend.create_session = AsyncMock(
            side_effect=OperationalError("stmt", {}, Exception("boom"))
        )
        with patch("quackmem.backend.get_backend", return_value=backend):
            @track_conversation(session_id=uuid.uuid4())
            async def my_llm_call(messages):
                return "result despite failure"

            result = await my_llm_call(["input"])

        assert result == "result despite failure"
