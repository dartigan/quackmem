from __future__ import annotations

from pydantic import BaseModel

from quackmem.schema.enums import MessageRole


class CanonicalMessage(BaseModel):
    role: MessageRole
    content: str | list[dict]
    tool_calls: list[dict] | None = None
    token_count: int | None = None
    metadata: dict | None = None
