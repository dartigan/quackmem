"""Tests for OpenAIAgentsWrapper and openai_agents_mem decorator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from quackmem.wrappers.openai_agents import OpenAIAgentsWrapper, openai_agents_mem
from quackmem.schema.enums import MessageRole


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

    def test_extracts_tool_calls_from_run_result(self):
        """A custom RunResult exposing ``tool_calls`` must surface them."""
        run_result = MagicMock()
        run_result.final_output = "thinking"
        run_result.tool_calls = [
            {"id": "c1", "name": "search"},
            {"id": "c2", "name": "fetch"},
        ]
        result = self.wrapper.extract_response(run_result)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2

    def test_run_result_without_tool_calls(self):
        run_result = _make_run_result("done")
        result = self.wrapper.extract_response(run_result)
        # _make_run_result sets only final_output / usage; no tool_calls attr or it's a Mock,
        # so the extractor must not surface a Mock as tool_calls.
        assert result.tool_calls is None or isinstance(result.tool_calls, list)

    def test_dict_result_with_tool_calls(self):
        result = self.wrapper.extract_response({
            "content": "x",
            "tool_calls": [{"id": "c1", "name": "t"}],
        })
        assert result.tool_calls == [{"id": "c1", "name": "t"}]


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
    @pytest.mark.asyncio
    async def test_wraps_async_function_correctly(self):
        backend = _mock_backend()
        run_result = _make_run_result("agent response", total_tokens=10)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @openai_agents_mem(session_id=uuid4())
            async def run_agent(input: str):
                return run_result

            result = await run_agent("what is the capital of France?")

        assert result is run_result
        backend.create_session.assert_called_once()
        # 1 input message inserted; assistant via reserve+finalize.
        assert backend.insert_message.call_count == 1
        backend.reserve_assistant_message.assert_called_once()
        backend.finalize_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_id_must_be_provided(self):
        """session_id is mandatory — wrapper raises ValueError if omitted."""
        backend = _mock_backend()

        with patch("quackmem.backend.get_backend", return_value=backend):
            @openai_agents_mem()
            async def run_agent(input: str):
                return _make_run_result("done")

            with pytest.raises(ValueError, match="session_id is required"):
                await run_agent("hello")

    @pytest.mark.asyncio
    async def test_session_id_from_kwargs_used_when_provided(self):
        """When session_id is provided, it is used."""
        fixed_session = uuid4()
        captured = {}
        backend = _mock_backend()

        async def capture_upsert(session):
            captured["session"] = session

        backend.create_session = capture_upsert

        with patch("quackmem.backend.get_backend", return_value=backend):
            @openai_agents_mem(session_id=fixed_session)
            async def run_agent(input: str):
                return _make_run_result("done")

            await run_agent("hello")

        assert captured["session"].id == fixed_session
