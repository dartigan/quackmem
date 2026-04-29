from __future__ import annotations
from typing import Any
import logging
from convo_tracker.schema.canonical import CanonicalMessage
from convo_tracker.schema.enums import MessageRole
from convo_tracker.wrappers.base import BaseWrapper
from convo_tracker.core.decorator import track

logger = logging.getLogger(__name__)

_ROLE_MAP = {
    "HumanMessage": MessageRole.user,
    "AIMessage": MessageRole.assistant,
    "SystemMessage": MessageRole.system,
    "ToolMessage": MessageRole.tool,
    "ChatMessage": MessageRole.user,
}


def _normalize_message(msg: Any) -> CanonicalMessage | None:
    """Normalize a LangChain BaseMessage or OpenAI dict to CanonicalMessage."""
    try:
        # LangChain BaseMessage
        class_name = type(msg).__name__
        if class_name in _ROLE_MAP:
            content = msg.content
            tool_calls = getattr(msg, "tool_calls", None) or None
            token_count = None
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                token_count = msg.usage_metadata.get("total_tokens")
            return CanonicalMessage(
                role=_ROLE_MAP[class_name],
                content=content,
                tool_calls=tool_calls,
                token_count=token_count,
            )
        # OpenAI-format dict
        if isinstance(msg, dict) and "role" in msg:
            return CanonicalMessage(
                role=MessageRole(msg["role"]),
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
            )
    except Exception:
        logger.debug("Could not normalize message: %s", msg, exc_info=True)
    return None


class LangGraphWrapper(BaseWrapper):

    def extract_messages(self, args: tuple, kwargs: dict) -> list[CanonicalMessage]:
        # LangGraph node: first arg is state dict
        state = args[0] if args else kwargs.get("state", {})
        if not isinstance(state, dict):
            return []
        raw_messages = state.get("messages", [])
        return [m for msg in raw_messages if (m := _normalize_message(msg)) is not None]

    def extract_response(self, result: Any) -> CanonicalMessage:
        # Result is typically {"messages": [AIMessage(...)]} or just a message
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                normalized = _normalize_message(last)
                if normalized:
                    return normalized
            # Fallback: stringify
            return CanonicalMessage(role=MessageRole.assistant, content=str(result))
        normalized = _normalize_message(result)
        if normalized:
            return normalized
        return CanonicalMessage(role=MessageRole.assistant, content=str(result))

    def extract_token_count(self, result: Any) -> int | None:
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                if hasattr(last, "usage_metadata") and last.usage_metadata:
                    return last.usage_metadata.get("total_tokens")
            # Check usage key directly
            usage = result.get("usage") or result.get("usage_metadata")
            if isinstance(usage, dict):
                return usage.get("total_tokens") or usage.get("output_tokens")
        return None


def langgraph_mem(**kwargs):
    """
    Decorator factory for LangGraph nodes.

    Usage:
        @langgraph_mem(user_id="u123", conversation_id=some_uuid)
        async def my_node(state: dict) -> dict:
            ...
    """
    wrapper = LangGraphWrapper()
    def decorator(fn):
        return track(wrapper, **kwargs)(fn)
    return decorator
