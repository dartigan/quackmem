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
    updated_at: datetime | None = None
    regeneration_count: int = 0
    metadata: dict = Field(default_factory=dict)


class MessageReservation(BaseModel):
    """Input to reserve an empty assistant message before generation runs.

    The decorator inserts this row with status=pending and an empty content
    string before invoking the wrapped function, so callers can observe a
    stable message_id mid-call.
    """

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    conversation_id: UUID
    parent_message_id: UUID | None = None
    role: MessageRole = MessageRole.assistant
    metadata: dict = Field(default_factory=dict)


class MessageFinalization(BaseModel):
    """Input to finalize a previously reserved message after generation.

    Called exactly once per reservation. On success, sets content/status to
    completed. On failure, sets status=failed and records the error string
    in metadata so the audit trail stays consistent.
    """

    model_config = ConfigDict(use_enum_values=True)

    message_id: UUID
    content: str | list[dict]
    token_count: int | None = None
    status: MessageStatus = MessageStatus.completed
    error: str | None = None
