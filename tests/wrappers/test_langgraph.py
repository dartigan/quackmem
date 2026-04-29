"""Tests for LangGraphWrapper and langgraph_mem decorator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from convo_tracker.wrappers.langgraph import LangGraphWrapper, langgraph_mem
from convo_tracker.schema.enums import MessageRole
from convo_tracker.core.decorator import init_decorator
from convo_tracker.core.config import TrackerConfig


# ---------------------------------------------------------------------------
# Mock LangChain message helpers
# ---------------------------------------------------------------------------

def _make_lc_message(class_name: str, content: str, usage_metadata: dict | None = None):
    """Create a mock LangChain-style message object."""
    msg = MagicMock()
    msg.__class__ = type(class_name, (), {})
    type(msg).__name__ = class_name
    msg.content = content
    msg.tool_calls = None
    if usage_metadata is not None:
        msg.usage_metadata = usage_metadata
    else:
        msg.usage_metadata = None
    return msg


# ---------------------------------------------------------------------------
# LangGraphWrapper.extract_messages
# ---------------------------------------------------------------------------

class TestLangGraphWrapperExtractMessages:
    def setup_method(self):
        self.wrapper = LangGraphWrapper()

    def test_extracts_human_message_from_state(self):
        human = _make_lc_message("HumanMessage", "hello")
        state = {"messages": [human]}
        result = self.wrapper.extract_messages((state,), {})
        assert len(result) == 1
        assert result[0].role == MessageRole.user
        assert result[0].content == "hello"

    def test_extracts_ai_message_from_state(self):
        ai = _make_lc_message("AIMessage", "I'm an AI")
        state = {"messages": [ai]}
        result = self.wrapper.extract_messages((state,), {})
        assert len(result) == 1
        assert result[0].role == MessageRole.assistant

    def test_extracts_multiple_messages(self):
        human = _make_lc_message("HumanMessage", "q")
        ai = _make_lc_message("AIMessage", "a")
        state = {"messages": [human, ai]}
        result = self.wrapper.extract_messages((state,), {})
        assert len(result) == 2

    def test_extracts_openai_format_dict(self):
        state = {"messages": [{"role": "user", "content": "hi"}]}
        result = self.wrapper.extract_messages((state,), {})
        assert len(result) == 1
        assert result[0].role == MessageRole.user
        assert result[0].content == "hi"

    def test_skips_unrecognized_message_type(self):
        weird = MagicMock()
        type(weird).__name__ = "WeirdMessage"
        state = {"messages": [weird, {"role": "user", "content": "real"}]}
        result = self.wrapper.extract_messages((state,), {})
        assert len(result) == 1
        assert result[0].content == "real"

    def test_empty_messages_list(self):
        state = {"messages": []}
        result = self.wrapper.extract_messages((state,), {})
        assert result == []

    def test_state_as_kwarg(self):
        human = _make_lc_message("HumanMessage", "from kwargs")
        state = {"messages": [human]}
        result = self.wrapper.extract_messages((), {"state": state})
        assert len(result) == 1

    def test_non_dict_state_returns_empty(self):
        result = self.wrapper.extract_messages(("not a dict",), {})
        assert result == []


# ---------------------------------------------------------------------------
# LangGraphWrapper.extract_response
# ---------------------------------------------------------------------------

class TestLangGraphWrapperExtractResponse:
    def setup_method(self):
        self.wrapper = LangGraphWrapper()

    def test_extracts_last_message_from_dict_result(self):
        ai = _make_lc_message("AIMessage", "final answer")
        result = self.wrapper.extract_response({"messages": [ai]})
        assert result.role == MessageRole.assistant
        assert result.content == "final answer"

    def test_fallback_stringifies_dict_without_messages(self):
        result = self.wrapper.extract_response({"something": "else"})
        assert result.role == MessageRole.assistant
        assert isinstance(result.content, str)

    def test_extracts_lc_message_directly(self):
        ai = _make_lc_message("AIMessage", "direct")
        result = self.wrapper.extract_response(ai)
        assert result.role == MessageRole.assistant
        assert result.content == "direct"

    def test_fallback_for_unknown_type(self):
        result = self.wrapper.extract_response(42)
        assert result.role == MessageRole.assistant
        assert "42" in result.content


# ---------------------------------------------------------------------------
# LangGraphWrapper.extract_token_count
# ---------------------------------------------------------------------------

class TestLangGraphWrapperExtractTokenCount:
    def setup_method(self):
        self.wrapper = LangGraphWrapper()

    def test_extracts_token_count_from_usage_metadata(self):
        ai = _make_lc_message("AIMessage", "reply", usage_metadata={"total_tokens": 99})
        result_dict = {"messages": [ai]}
        count = self.wrapper.extract_token_count(result_dict)
        assert count == 99

    def test_returns_none_when_no_usage(self):
        ai = _make_lc_message("AIMessage", "reply")
        count = self.wrapper.extract_token_count({"messages": [ai]})
        assert count is None

    def test_extracts_from_top_level_usage_key(self):
        count = self.wrapper.extract_token_count({"usage": {"total_tokens": 50}})
        assert count == 50

    def test_extracts_from_usage_metadata_key(self):
        count = self.wrapper.extract_token_count({"usage_metadata": {"total_tokens": 25}})
        assert count == 25

    def test_returns_none_for_non_dict_result(self):
        count = self.wrapper.extract_token_count("string result")
        assert count is None


# ---------------------------------------------------------------------------
# langgraph_mem decorator
# ---------------------------------------------------------------------------

class TestLangGraphMemDecorator:
    @pytest.fixture(autouse=True)
    def setup_config(self):
        cfg = TrackerConfig(
            database_url="postgresql+asyncpg://x:x@localhost/x",
            sync_mode=True,
        )
        init_decorator(cfg)

    @pytest.mark.asyncio
    async def test_wraps_async_function(self):
        write_mock = AsyncMock(return_value=None)
        with patch("convo_tracker.worker.tasks.sync_write_message", write_mock):
            @langgraph_mem()
            async def my_node(state: dict) -> dict:
                return {"messages": [_make_lc_message("AIMessage", "wrapped result")]}

            state = {"messages": [_make_lc_message("HumanMessage", "q")]}
            result = await my_node(state)

        assert result["messages"][0].content == "wrapped result"
        write_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_track(self):
        write_mock = AsyncMock(return_value=None)
        captured = {}

        async def capture_write(session, message):
            captured["session"] = session

        with patch("convo_tracker.worker.tasks.sync_write_message", capture_write):
            fixed_conv = uuid4()

            @langgraph_mem(conversation_id=fixed_conv)
            async def my_node(state: dict) -> dict:
                return {"messages": [_make_lc_message("AIMessage", "ok")]}

            await my_node({"messages": []})

        assert captured["session"].conversation_id == fixed_conv
