from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from uuid import UUID


@dataclass
class TrackingContext:
    session_id: UUID
    conversation_id: UUID
    message_id: UUID | None = None
    parent_message_id: UUID | None = None


_tracking_ctx: ContextVar[TrackingContext | None] = ContextVar(
    "_tracking_ctx", default=None
)


def set_tracking_context(ctx: TrackingContext) -> Token:
    """Set the active tracking context. Returns the token for reset."""
    return _tracking_ctx.set(ctx)


def get_tracking_context() -> TrackingContext | None:
    """Return the current tracking context or None if not in a tracked call."""
    return _tracking_ctx.get()


def reset_tracking_context(token: Token) -> None:
    """Reset to the previous context state using the token from set_tracking_context."""
    _tracking_ctx.reset(token)
