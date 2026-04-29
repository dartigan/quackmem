"""Tests for OpenAIAgentsWrapper and openai_agents_mem decorator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from convo_tracker.wrappers.openai_agents import OpenAIAgentsWrapper, openai_agents_mem
from convo_tracker.schema.enums import MessageRole
from convo_tracker.core.decorator import init_decorator
from convo_tracker.core.config import TrackerConfig


# ---------------------------------------------------------------------------
# Mock RunResult
# ---------------------------------------------------------------------------

def _make_run_result(final_output: str, total_tokens: int | None = None):
    result = MagicMock()
    result.final_output = final_output
    if total_tokens is not None:
        usage = MagicMock()
        usage.total_tokens = total_tokens
        result.usage = usage
    else:
        result.usage = None
    return result


# ---------------------------------------------------------------------------
# OpenAIAgentsWrapper.extract_messages
# ---------------------------------------------------------------------------

class TestOpenAIAgentsWrapperExtractMessages:
    def setup_method(self):
        self.wrapper = OpenAIAgentsWrapper()

    def test_string_input_becomes_user_message(self):
        result = self.wrapper.extract_messages(("hello world",), {})
        assert len(result) == 1
        assert result[0].role == MessageRole.user
        assert result[0].content == "hello world"

    def test_list_of_dicts_input(self):
        messages = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
        ]
        result = self.wrapper.extract_messages((messages,), {})
        assert len(result) == 2
        assert result[0].role == MessageRole.user
        assert result[1].role == MessageRole.assistant

    def test_input_kwarg_fallback(self):
        result = self.wrapper.extract_messages((), {"input": "from kwarg"})
        assert len(result) == 1
        assert result[0].content == "from kwarg"

    def test_messages_kwarg_fallback(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = self.wrapper.extract_messages((), {"messages": msgs})
        assert len(result) == 1

    def test_invalid_role_skipped(self):
        messages = [
            {"role": "user", "content": "valid"},
            {"role": "not_a_role", "content": "invalid"},
        ]
        result = self.wrapper.extract_messages((messages,), {})
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        result = self.wrapper.extract_messages(([], ), {})
        assert result == []

    def test_tool_calls_extracted(self):
        calls = [{"id": "c1", "type": "function"}]
        messages = [{"role": "assistant", "content": "", "tool_calls": calls}]
        result = self.wrapper.extract_messages((messages,), {})
        assert result[0].tool_calls == calls


# ---------------------------------------------------------------------------
# OpenAIAgentsWrapper.extract_response
# ---------------------------------------------------------------------------

class TestOpenAIAgentsWrapperExtractResponse:
    def setup_method(self):
        self.wrapper = OpenAIAgentsWrapper()

    def test_extracts_final_output_from_run_result(self):
        run_result = _make_run_result("the final answer")
        result = self.wrapper.extract_response(run_result)
        assert result.role == MessageRole.assistant
        assert result.content == "the final answer"

    def test_plain_string_result(self):
        result = self.wrapper.extract_response("plain text")
        assert result.role == MessageRole.assistant
        assert result.content == "plain text"

    def test_dict_result_fallback(self):
        result = self.wrapper.extract_response({"some": "dict"})
        assert result.role == MessageRole.assistant
        assert isinstance(result.content, str)

    def test_non_string_final_output_stringified(self):
        run_result = MagicMock()
        run_result.final_output = {"complex": "output"}
        result = self.wrapper.extract_response(run_result)
        assert isinstance(result.content, str)


# ---------------------------------------------------------------------------
# OpenAIAgentsWrapper.extract_token_count
# ---------------------------------------------------------------------------

class TestOpenAIAgentsWrapperExtractTokenCount:
    def setup_method(self):
        self.wrapper = OpenAIAgentsWrapper()

    def test_extracts_total_tokens_from_usage(self):
        run_result = _make_run_result("ok", total_tokens=150)
        count = self.wrapper.extract_token_count(run_result)
        assert count == 150

    def test_returns_none_when_no_usage(self):
        run_result = _make_run_result("ok", total_tokens=None)
        count = self.wrapper.extract_token_count(run_result)
        assert count is None

    def test_returns_none_for_string_result(self):
        count = self.wrapper.extract_token_count("plain string")
        assert count is None


# ---------------------------------------------------------------------------
# openai_agents_mem decorator
# ---------------------------------------------------------------------------

class TestOpenAIAgentsMemDecorator:
    @pytest.fixture(autouse=True)
    def setup_config(self):
        cfg = TrackerConfig(
            database_url="postgresql+asyncpg://x:x@localhost/x",
            sync_mode=True,
        )
        init_decorator(cfg)

    @pytest.mark.asyncio
    async def test_wraps_async_function_correctly(self):
        write_mock = AsyncMock(return_value=None)
        run_result = _make_run_result("agent response", total_tokens=10)

        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @openai_agents_mem()
            async def run_agent(input: str):
                return run_result

            result = await run_agent("what is the capital of France?")

        assert result is run_result
        write_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_session_id_to_track(self):
        write_mock = AsyncMock(return_value=None)
        captured = {}
        fixed_session = uuid4()

        async def capture_write(session, message):
            captured["session"] = session

        with patch("convo_tracker.worker.tasks.sync_write_message", capture_write):
            @openai_agents_mem(session_id=fixed_session)
            async def run_agent(input: str):
                return _make_run_result("done")

            await run_agent("hello")

        assert captured["session"].id == fixed_session
