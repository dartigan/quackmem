"""Tests for GenericWrapper and track_conversation decorator."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from convo_tracker.wrappers.generic import GenericWrapper, track_conversation
from convo_tracker.schema.enums import MessageRole
from convo_tracker.core.decorator import init_decorator
from convo_tracker.core.config import TrackerConfig


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
    @pytest.fixture(autouse=True)
    def setup_config(self):
        cfg = TrackerConfig(
            database_url="postgresql+asyncpg://x:x@localhost/x",
            sync_mode=True,
        )
        init_decorator(cfg)

    @pytest.mark.asyncio
    async def test_async_function_wrapped(self):
        write_mock = AsyncMock(return_value=None)
        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track_conversation()
            async def my_llm_call(messages):
                return "llm output"

            result = await my_llm_call([{"role": "user", "content": "test"}])

        assert result == "llm output"
        write_mock.assert_called_once()

    def test_sync_function_wrapped(self):
        write_mock = AsyncMock(return_value=None)
        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @track_conversation()
            def my_sync_call(messages):
                return "sync output"

            result = my_sync_call([{"role": "user", "content": "hello"}])

        assert result == "sync output"

    @pytest.mark.asyncio
    async def test_custom_metadata_passed(self):
        captured = {}
        write_mock = AsyncMock(return_value=None)

        async def capture_write(session, message):
            captured["session"] = session

        with patch("convo_tracker.worker.tasks.sync_write_message", capture_write):
            @track_conversation(user_id="bob")
            async def my_llm_call(messages):
                return "ok"

            await my_llm_call(["hi"])

        assert captured["session"].metadata.get("user_id") == "bob"

    @pytest.mark.asyncio
    async def test_tracking_error_does_not_break_caller(self):
        failing_write = AsyncMock(side_effect=Exception("boom"))
        with patch("convo_tracker.worker.tasks.sync_write_message", failing_write):
            @track_conversation()
            async def my_llm_call(messages):
                return "result despite failure"

            result = await my_llm_call(["input"])

        assert result == "result despite failure"
