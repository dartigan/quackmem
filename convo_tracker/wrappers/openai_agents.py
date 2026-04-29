from __future__ import annotations
from typing import Any
import logging
from convo_tracker.schema.canonical import CanonicalMessage
from convo_tracker.schema.enums import MessageRole
from convo_tracker.wrappers.base import BaseWrapper
from convo_tracker.core.decorator import track

logger = logging.getLogger(__name__)


class OpenAIAgentsWrapper(BaseWrapper):

    def extract_messages(self, args: tuple, kwargs: dict) -> list[CanonicalMessage]:
        # OpenAI Agents SDK: input is typically a list of dicts or a string
        # First positional arg or 'input' kwarg
        messages_input = args[0] if args else kwargs.get("input", kwargs.get("messages", []))
        if isinstance(messages_input, str):
            return [CanonicalMessage(role=MessageRole.user, content=messages_input)]
        if isinstance(messages_input, list):
            result = []
            for msg in messages_input:
                if isinstance(msg, dict) and "role" in msg:
                    try:
                        result.append(CanonicalMessage(
                            role=MessageRole(msg["role"]),
                            content=msg.get("content", ""),
                            tool_calls=msg.get("tool_calls"),
                        ))
                    except Exception:
                        logger.debug("Could not parse message: %s", msg)
            return result
        return []

    def extract_response(self, result: Any) -> CanonicalMessage:
        # RunResult has .final_output attribute
        if hasattr(result, "final_output"):
            content = result.final_output
            if isinstance(content, str):
                return CanonicalMessage(role=MessageRole.assistant, content=content)
            return CanonicalMessage(role=MessageRole.assistant, content=str(content))
        # Plain string or dict fallback
        if isinstance(result, str):
            return CanonicalMessage(role=MessageRole.assistant, content=result)
        return CanonicalMessage(role=MessageRole.assistant, content=str(result))

    def extract_token_count(self, result: Any) -> int | None:
        # RunResult.usage.total_tokens
        try:
            usage = getattr(result, "usage", None)
            if usage is not None:
                return getattr(usage, "total_tokens", None)
        except Exception:
            pass
        return None


def openai_agents_mem(**kwargs):
    """
    Decorator factory for OpenAI Agents SDK runner functions.

    Usage:
        @openai_agents_mem(user_id="u123")
        async def run_agent(input: str) -> RunResult:
            ...
    """
    wrapper = OpenAIAgentsWrapper()
    def decorator(fn):
        return track(wrapper, **kwargs)(fn)
    return decorator
