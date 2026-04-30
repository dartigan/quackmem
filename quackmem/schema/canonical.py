from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from quackmem.schema.enums import MessageRole


class ToolCall(BaseModel):
    """A single tool/function call emitted by an assistant message.

    Permissive shape — fields not listed here pass through via
    ``extra="allow"``. This keeps the model framework-agnostic: OpenAI's
    nested ``function: {name, arguments}`` block, Anthropic's ``input``
    field, and Google ADK's ``function_call`` structure all round-trip
    losslessly.

    Standard fields:
        id: Tool-call identifier (OpenAI ``id``, Anthropic ``tool_use_id``,
            ADK call id). Used to link a ``role=tool`` result row back to
            this call.
        name: Tool/function name. ``None`` is allowed because some
            frameworks (notably OpenAI's chat-completions format) nest the
            name under a ``function`` block instead of a top-level field.
        arguments: Call arguments. OpenAI emits a JSON string; Anthropic
            and ADK emit a dict. Quackmem stores whatever it receives.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    arguments: dict | str | None = None


class CanonicalMessage(BaseModel):
    role: MessageRole
    content: str | list[dict]
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    token_count: int | None = None
    metadata: dict | None = None
