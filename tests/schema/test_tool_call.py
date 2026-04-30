"""Tests for the ToolCall Pydantic model and tool-call schema fields."""
from __future__ import annotations

from uuid import uuid4

from quackmem.schema.canonical import CanonicalMessage, ToolCall
from quackmem.schema.enums import MessageRole
from quackmem.schema.models import (
    MessageFinalization,
    MessageReservation,
    TrackedMessage,
)


# ---------------------------------------------------------------------------
# ToolCall — permissive, framework-neutral shape
# ---------------------------------------------------------------------------

class TestToolCall:
    def test_all_fields_optional(self):
        """A bare ToolCall() is valid — id/name/arguments are all optional."""
        call = ToolCall()
        assert call.id is None
        assert call.name is None
        assert call.arguments is None

    def test_openai_chat_completions_shape_passes_through_extras(self):
        """OpenAI nests ``name`` under a ``function`` block — must round-trip."""
        raw = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
        }
        call = ToolCall(**raw)
        # Top-level id is captured, rest passes through via extra="allow".
        assert call.id == "call_abc"
        # The nested function block should still be readable on the model.
        assert call.model_dump()["function"]["name"] == "get_weather"
        assert call.model_dump()["type"] == "function"

    def test_anthropic_shape_with_dict_arguments(self):
        call = ToolCall(id="toolu_xyz", name="get_weather", arguments={"city": "SF"})
        assert call.id == "toolu_xyz"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "SF"}

    def test_openai_string_arguments_accepted(self):
        """OpenAI emits arguments as a JSON string — must NOT be coerced to dict."""
        call = ToolCall(name="get_weather", arguments='{"city":"SF"}')
        assert call.arguments == '{"city":"SF"}'

    def test_google_adk_function_call_shape(self):
        """ADK uses ``args`` instead of ``arguments``; that key passes through."""
        call = ToolCall(name="get_weather", args={"city": "SF"})
        assert call.name == "get_weather"
        # ``args`` is preserved via extras even though it isn't a typed field.
        assert call.model_dump()["args"] == {"city": "SF"}

    def test_round_trips_through_model_dump(self):
        original = {
            "id": "call_1",
            "name": "tool",
            "arguments": {"x": 1},
            "custom_field": "preserved",
        }
        call = ToolCall(**original)
        dumped = call.model_dump()
        assert dumped["custom_field"] == "preserved"
        # Re-validation must succeed on the dumped form.
        ToolCall(**dumped)


# ---------------------------------------------------------------------------
# CanonicalMessage gains tool_calls / tool_call_id
# ---------------------------------------------------------------------------

class TestCanonicalMessageToolFields:
    def test_tool_calls_default_none(self):
        msg = CanonicalMessage(role=MessageRole.assistant, content="hi")
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_tool_calls_accepts_list_of_dicts(self):
        calls = [{"id": "c1", "name": "t"}, {"id": "c2", "name": "u"}]
        msg = CanonicalMessage(
            role=MessageRole.assistant, content="", tool_calls=calls,
        )
        assert msg.tool_calls == calls

    def test_tool_call_id_on_role_tool(self):
        msg = CanonicalMessage(
            role=MessageRole.tool,
            content="result",
            tool_call_id="c1",
        )
        assert msg.tool_call_id == "c1"


# ---------------------------------------------------------------------------
# Storage models — TrackedMessage / MessageReservation / MessageFinalization
# ---------------------------------------------------------------------------

class TestTrackedMessageToolFields:
    def test_defaults_are_none(self):
        m = TrackedMessage(
            session_id=uuid4(), conversation_id=uuid4(),
            role=MessageRole.assistant, content="hi",
        )
        assert m.tool_calls is None
        assert m.tool_call_id is None

    def test_assistant_row_with_multi_tool_calls(self):
        calls = [
            {"id": "c1", "name": "search"},
            {"id": "c2", "name": "fetch"},
            {"id": "c3", "name": "summarise"},
        ]
        m = TrackedMessage(
            session_id=uuid4(), conversation_id=uuid4(),
            role=MessageRole.assistant, content="", tool_calls=calls,
        )
        assert len(m.tool_calls) == 3
        assert {c["name"] for c in m.tool_calls} == {"search", "fetch", "summarise"}

    def test_tool_row_with_tool_call_id(self):
        m = TrackedMessage(
            session_id=uuid4(), conversation_id=uuid4(),
            role=MessageRole.tool, content="42", tool_call_id="c1",
        )
        assert m.tool_call_id == "c1"


class TestMessageFinalizationToolCalls:
    def test_default_none(self):
        f = MessageFinalization(message_id=uuid4(), content="done")
        assert f.tool_calls is None

    def test_accepts_calls(self):
        calls = [{"id": "c1", "name": "search"}]
        f = MessageFinalization(message_id=uuid4(), content="", tool_calls=calls)
        assert f.tool_calls == calls


class TestMessageReservationToolCallId:
    def test_default_none(self):
        r = MessageReservation(session_id=uuid4(), conversation_id=uuid4())
        assert r.tool_call_id is None

    def test_accepts_tool_call_id(self):
        r = MessageReservation(
            session_id=uuid4(), conversation_id=uuid4(),
            role=MessageRole.tool, tool_call_id="c1",
        )
        assert r.tool_call_id == "c1"
