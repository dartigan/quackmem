from __future__ import annotations
from typing import Any
import logging
from quackmem.schema.canonical import CanonicalMessage
from quackmem.schema.enums import MessageRole
from quackmem.wrappers.base import BaseWrapper
from quackmem.core.decorator import track

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
            tool_calls = self._extract_response_tool_calls(result)
            if isinstance(content, str):
                return CanonicalMessage(
                    role=MessageRole.assistant,
                    content=content,
                    tool_calls=tool_calls,
                )
            return CanonicalMessage(
                role=MessageRole.assistant,
                content=str(content),
                tool_calls=tool_calls,
            )
        # Plain string or dict fallback
        if isinstance(result, str):
            return CanonicalMessage(role=MessageRole.assistant, content=result)
        if isinstance(result, dict):
            calls = result.get("tool_calls")
            if calls is not None and not isinstance(calls, list):
                calls = None
            return CanonicalMessage(
                role=MessageRole.assistant,
                content=str(result.get("content", result)),
                tool_calls=calls,
            )
        return CanonicalMessage(role=MessageRole.assistant, content=str(result))

    @staticmethod
    def _extract_response_tool_calls(result: Any) -> list[dict] | None:
        """Best-effort pull of pending tool calls off a RunResult.

        The OpenAI Agents SDK normally resolves tool calls inside the run
        loop, so a RunResult typically has none outstanding. When a custom
        runner exposes them (e.g. via ``result.tool_calls`` or via the last
        item of ``result.new_items``), we pass them through verbatim.
        """
        try:
            calls = getattr(result, "tool_calls", None)
            if isinstance(calls, list):
                return calls
        except (AttributeError, TypeError) as exc:
            logger.debug("Could not read tool_calls off %r: %s", type(result).__name__, exc)
        return None

    def extract_token_count(self, result: Any) -> int | None:
        # RunResult.usage.total_tokens — guard against descriptor/attr errors
        # raised by exotic result objects.
        try:
            usage = getattr(result, "usage", None)
            if usage is not None:
                return getattr(usage, "total_tokens", None)
        except (AttributeError, TypeError) as exc:
            logger.debug("Could not extract token count from %r: %s", type(result).__name__, exc)
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
