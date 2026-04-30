from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from quackmem.wrappers.base import BaseWrapper

from quackmem.core.context import (
    TrackingContext,
    set_tracking_context,
    get_tracking_context,
    reset_tracking_context,
    save_last_tracking_context,
)
from quackmem.core.registry import validate_metadata
from quackmem.core.tasks import register_pending_task

from quackmem.schema.models import (
    TrackedSession,
    TrackedMessage,
    MessageReservation,
    MessageFinalization,
)
from quackmem.schema.enums import MessageStatus

logger = logging.getLogger(__name__)

# Errors expected from the storage layer. Any other exception type from
# _fire_write indicates a bug in quackmem itself or in user-supplied wrapper
# code, and must propagate so the caller sees it.
_EXPECTED_WRITE_ERRORS: tuple[type[BaseException], ...] = (
    SQLAlchemyError,
    OSError,
    asyncio.TimeoutError,
)

_RESERVED_KEYS = frozenset({"conversation_id", "session_id", "regenerate_message_id"})


def _resolve_ids(decorator_kwargs: dict) -> tuple[uuid.UUID, uuid.UUID]:
    session_id = decorator_kwargs.get("session_id")
    if session_id is None:
        session_id = uuid.uuid4()
    elif not isinstance(session_id, uuid.UUID):
        session_id = uuid.UUID(str(session_id))

    conversation_id = decorator_kwargs.get("conversation_id")
    if conversation_id is None:
        conversation_id = uuid.uuid4()
    elif not isinstance(conversation_id, uuid.UUID):
        conversation_id = uuid.UUID(str(conversation_id))
    return session_id, conversation_id


def _extract_metadata(decorator_kwargs: dict) -> dict:
    return {k: v for k, v in decorator_kwargs.items() if k not in _RESERVED_KEYS}


async def _pre_write(
    wrapper: BaseWrapper,
    decorator_kwargs: dict,
    args: tuple,
    fn_kwargs: dict,
    session_id: uuid.UUID,
    conversation_id: uuid.UUID,
    metadata: dict,
) -> uuid.UUID | None:
    """Persist session, dedup+insert input messages, reserve an empty
    assistant message. Returns the reserved assistant message id, or None
    when running in regeneration mode (no new reservation needed).

    Updates the active TrackingContext so the wrapped function can observe
    its own message_id mid-call.
    """
    from quackmem.backend import get_backend
    backend = get_backend()

    session_model = TrackedSession(
        id=session_id,
        conversation_id=conversation_id,
        metadata=metadata,
    )
    await backend.create_session(session_model)

    messages = wrapper.extract_messages(args, fn_kwargs)
    existing_rows = await backend.get_messages(session_id)

    def _normalize_content(content: Any) -> Any:
        if isinstance(content, str):
            return " ".join(content.split())
        return content

    def _message_key(msg: Any) -> tuple[str, Any]:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        return (role, _normalize_content(msg.content))

    existing_keys = [
        (str(row["role"]), _normalize_content(row["content"])) for row in existing_rows
    ]
    incoming_keys = [_message_key(msg) for msg in messages]

    prefix_len = 0
    for existing_key, incoming_key in zip(existing_keys, incoming_keys):
        if existing_key == incoming_key:
            prefix_len += 1
        else:
            break

    last_input_message_id = None
    if prefix_len > 0 and existing_rows:
        last_input_message_id = uuid.UUID(str(existing_rows[prefix_len - 1]["id"]))

    last_role = None
    last_content = None
    if prefix_len > 0:
        last_role, last_content = existing_keys[prefix_len - 1]

    regenerate_message_id = decorator_kwargs.get("regenerate_message_id")

    if not regenerate_message_id:
        for msg in messages[prefix_len:]:
            role, content = _message_key(msg)
            if role == last_role and content == last_content:
                continue
            msg_id = uuid.uuid4()
            await backend.insert_message(TrackedMessage(
                id=msg_id,
                session_id=session_id,
                conversation_id=conversation_id,
                parent_message_id=last_input_message_id,
                role=msg.role,
                content=msg.content,
                token_count=msg.token_count,
                status=MessageStatus.completed,
                metadata=msg.metadata or {},
            ))
            last_input_message_id = msg_id
            last_role, last_content = role, content

    if regenerate_message_id:
        regen_id = uuid.UUID(str(regenerate_message_id))
        ctx = get_tracking_context()
        if ctx is not None:
            ctx.message_id = regen_id
            ctx.parent_message_id = last_input_message_id
            logger.debug(
                "TrackingContext updated (regeneration)",
                extra={
                    "session_id": str(ctx.session_id),
                    "message_id": str(regen_id),
                },
            )
        return None

    reservation = MessageReservation(
        session_id=session_id,
        conversation_id=conversation_id,
        parent_message_id=last_input_message_id,
    )
    await backend.reserve_assistant_message(reservation)

    ctx = get_tracking_context()
    if ctx is not None:
        ctx.message_id = reservation.id
        ctx.parent_message_id = last_input_message_id
        logger.debug(
            "TrackingContext updated (reservation)",
            extra={
                "session_id": str(ctx.session_id),
                "message_id": str(reservation.id),
                "parent_message_id": (
                    str(last_input_message_id) if last_input_message_id else None
                ),
            },
        )

    return reservation.id


async def _post_write(
    wrapper: BaseWrapper,
    decorator_kwargs: dict,
    result: Any,
    reserved_message_id: uuid.UUID | None,
    error: BaseException | None = None,
) -> None:
    """Finalize the assistant message reserved by ``_pre_write``.

    On success, sets content + status=completed. On error, sets status=failed
    and records the error string in metadata. Regeneration falls through to
    ``update_message`` so the existing ``regeneration_count`` semantics are
    preserved.
    """
    from quackmem.backend import get_backend
    backend = get_backend()
    regenerate_message_id = decorator_kwargs.get("regenerate_message_id")

    if regenerate_message_id:
        if error is not None:
            return
        response = wrapper.extract_response(result)
        regen_id = uuid.UUID(str(regenerate_message_id))
        await backend.update_message(
            regen_id,
            response.content,
            regeneration_count=None,
        )
        return

    if reserved_message_id is None:
        return

    if error is not None:
        await backend.finalize_message(MessageFinalization(
            message_id=reserved_message_id,
            content="",
            status=MessageStatus.failed,
            error=f"{type(error).__name__}: {error}",
        ))
        return

    response = wrapper.extract_response(result)
    token_count = wrapper.extract_token_count(result) or response.token_count
    await backend.finalize_message(MessageFinalization(
        message_id=reserved_message_id,
        content=response.content,
        token_count=token_count,
        status=MessageStatus.completed,
    ))


async def _safe_pre_write(
    wrapper, decorator_kwargs, args, fn_kwargs,
    session_id, conversation_id, metadata, *, path: str,
) -> uuid.UUID | None:
    try:
        return await _pre_write(
            wrapper, decorator_kwargs, args, fn_kwargs,
            session_id, conversation_id, metadata,
        )
    except _EXPECTED_WRITE_ERRORS as exc:
        logger.error(
            "Tracking pre-write failed",
            exc_info=True,
            extra={
                "session_id": str(session_id),
                "conversation_id": str(conversation_id),
                "error_type": type(exc).__name__,
                "path": path,
            },
        )
        return None


async def _safe_post_write(
    wrapper, decorator_kwargs, result, reserved_id,
    *, path: str, error: BaseException | None = None,
) -> None:
    try:
        await _post_write(wrapper, decorator_kwargs, result, reserved_id, error=error)
    except _EXPECTED_WRITE_ERRORS as exc:
        logger.error(
            "Tracking post-write failed",
            exc_info=True,
            extra={
                "error_type": type(exc).__name__,
                "path": path,
            },
        )


async def _run_tracked_async(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)
    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)
    reserved_id: uuid.UUID | None = None
    try:
        reserved_id = await _safe_pre_write(
            wrapper, decorator_kwargs, args, fn_kwargs,
            session_id, conversation_id, metadata, path="async",
        )

        try:
            result = await fn(*args, **fn_kwargs)
        except BaseException as exc:
            await _safe_post_write(
                wrapper, decorator_kwargs, None, reserved_id,
                path="async", error=exc,
            )
            raise

        await _safe_post_write(
            wrapper, decorator_kwargs, result, reserved_id, path="async",
        )
        return result
    finally:
        save_last_tracking_context(ctx)
        reset_tracking_context(token)


async def _run_tracked_asyncgen(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)
    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)
    reserved_id: uuid.UUID | None = None
    chunks: list = []
    try:
        reserved_id = await _safe_pre_write(
            wrapper, decorator_kwargs, args, fn_kwargs,
            session_id, conversation_id, metadata, path="asyncgen",
        )

        try:
            async for chunk in fn(*args, **fn_kwargs):
                chunks.append(chunk)
                yield chunk
        except BaseException as exc:
            await _safe_post_write(
                wrapper, decorator_kwargs, chunks, reserved_id,
                path="asyncgen", error=exc,
            )
            raise

        await _safe_post_write(
            wrapper, decorator_kwargs, chunks, reserved_id, path="asyncgen",
        )
    finally:
        save_last_tracking_context(ctx)
        reset_tracking_context(token)


def _run_tracked_sync(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)
    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    reserved_id: uuid.UUID | None = None

    try:
        # Pre-write must run before fn(). With a running loop we can't block,
        # so degrade to legacy fire-and-forget single-shot at the end (no
        # mid-call message_id, but the function still works).
        if running_loop is None:
            reserved_id = asyncio.run(_safe_pre_write(
                wrapper, decorator_kwargs, args, fn_kwargs,
                session_id, conversation_id, metadata, path="sync",
            ))

        try:
            result = fn(*args, **fn_kwargs)
        except BaseException as exc:
            if running_loop is None and reserved_id is not None:
                asyncio.run(_safe_post_write(
                    wrapper, decorator_kwargs, None, reserved_id,
                    path="sync", error=exc,
                ))
            raise

        if running_loop is None:
            asyncio.run(_safe_post_write(
                wrapper, decorator_kwargs, result, reserved_id, path="sync",
            ))
        else:
            # Running loop: schedule a single-shot guarded write that does
            # both pre- and post-work in the background.
            coro = _guarded_full_write(
                wrapper, decorator_kwargs, args, fn_kwargs, result,
                session_id, conversation_id, metadata,
            )
            register_pending_task(running_loop.create_task(coro))

        return result
    finally:
        save_last_tracking_context(ctx)
        reset_tracking_context(token)


async def _guarded_full_write(
    wrapper: BaseWrapper,
    decorator_kwargs: dict,
    args: tuple,
    fn_kwargs: dict,
    result: Any,
    session_id: uuid.UUID,
    conversation_id: uuid.UUID,
    metadata: dict,
) -> None:
    """Background fallback for sync-in-running-loop callers. Performs the
    full pre+post write sequence after the function has already returned.
    The caller's TrackingContext won't see message_id mid-call here, but the
    persisted state still ends up identical to the foreground path."""
    try:
        reserved_id = await _pre_write(
            wrapper, decorator_kwargs, args, fn_kwargs,
            session_id, conversation_id, metadata,
        )
    except _EXPECTED_WRITE_ERRORS as exc:
        logger.error(
            "Tracking pre-write failed (background task)",
            exc_info=True,
            extra={
                "session_id": str(session_id),
                "error_type": type(exc).__name__,
                "path": "sync_background",
            },
        )
        return

    try:
        await _post_write(wrapper, decorator_kwargs, result, reserved_id)
    except _EXPECTED_WRITE_ERRORS as exc:
        logger.error(
            "Tracking post-write failed (background task)",
            exc_info=True,
            extra={
                "session_id": str(session_id),
                "error_type": type(exc).__name__,
                "path": "sync_background",
            },
        )


def track(wrapper: BaseWrapper, **decorator_kwargs):
    """
    Decorator factory. Usage:

        @track(MyWrapper(), user_id="abc")
        async def my_agent(state): ...

    OR used as a factory by wrapper-specific decorators:

        def langgraph_mem(**kwargs):
            def decorator(fn):
                return track(LangGraphWrapper(), **kwargs)(fn)
            return decorator
    """
    def decorator(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                return await _run_tracked_async(fn, wrapper, decorator_kwargs, args, kwargs)
            return async_wrapper
        elif inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            async def asyncgen_wrapper(*args, **kwargs):
                async for chunk in _run_tracked_asyncgen(fn, wrapper, decorator_kwargs, args, kwargs):
                    yield chunk
            return asyncgen_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                return _run_tracked_sync(fn, wrapper, decorator_kwargs, args, kwargs)
            return sync_wrapper

    return decorator
