from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from quackmem.schema.enums import MessageRole, MessageStatus


class TrackedSession(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = Field(default_factory=dict)


class TrackedMessage(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    conversation_id: UUID
    parent_message_id: UUID | None = None
    role: MessageRole
    content: str | list[dict]
    token_count: int | None = None
    status: MessageStatus = MessageStatus.completed
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = Field(default_factory=dict)
