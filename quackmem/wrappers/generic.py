from __future__ import annotations
from typing import Any
import logging
from quackmem.schema.canonical import CanonicalMessage
from quackmem.schema.enums import MessageRole
from quackmem.wrappers.base import BaseWrapper
from quackmem.core.decorator import track

logger = logging.getLogger(__name__)


class GenericWrapper(BaseWrapper):

    def extract_messages(self, args: tuple, kwargs: dict) -> list[CanonicalMessage]:
        messages_input = args[0] if args else kwargs.get("messages", kwargs.get("input", []))
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
        if isinstance(result, str):
            return CanonicalMessage(role=MessageRole.assistant, content=result)
        if isinstance(result, dict):
            raw = result.get("content", result.get("output", str(result)))
            content = raw if isinstance(raw, (str, list)) else str(raw)
            return CanonicalMessage(role=MessageRole.assistant, content=content)
        return CanonicalMessage(role=MessageRole.assistant, content=str(result))

    def extract_token_count(self, result: Any) -> int | None:
        if isinstance(result, dict):
            usage = result.get("usage")
            if isinstance(usage, dict):
                return usage.get("total_tokens") or usage.get("completion_tokens")
        return None


def track_conversation(**kwargs):
    """
    Generic decorator factory. Works with any function that takes OpenAI-format
    message dicts as the first argument and returns a string or dict.

    Usage:
        @track_conversation(user_id="u123", sync_mode note: set in TrackerConfig)
        async def my_llm_call(messages: list[dict]) -> str:
            ...
    """
    wrapper = GenericWrapper()
    def decorator(fn):
        return track(wrapper, **kwargs)(fn)
    return decorator
