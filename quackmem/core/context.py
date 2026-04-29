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


# Active during a decorated call — reset to None (or outer context) once the
# call returns.
_tracking_ctx: ContextVar[TrackingContext | None] = ContextVar(
    "_tracking_ctx", default=None
)

# Set to the most recent context *after* a decorated call finishes so that
# callers can read session_id / message_id once the function has returned.
_last_tracking_ctx: ContextVar[TrackingContext | None] = ContextVar(
    "_last_tracking_ctx", default=None
)


def set_tracking_context(ctx: TrackingContext) -> Token:
    """Set the active tracking context. Returns the token for reset."""
    return _tracking_ctx.set(ctx)


def get_tracking_context() -> TrackingContext | None:
    """Return the tracking context for the current or most-recent tracked call.

    During a decorated call this returns the *active* context.
    After a decorated call returns, this returns the context of the last
    completed call so callers can inspect session_id, conversation_id, and
    message_id.
    """
    active = _tracking_ctx.get()
    if active is not None:
        return active
    return _last_tracking_ctx.get()


def save_last_tracking_context(ctx: TrackingContext) -> None:
    """Persist *ctx* as the last completed tracking context for this task."""
    _last_tracking_ctx.set(ctx)


def reset_tracking_context(token: Token) -> None:
    """Reset the active context to the previous state using the token from set_tracking_context."""
    _tracking_ctx.reset(token)
