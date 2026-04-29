from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import uuid
from typing import TYPE_CHECKING, Any

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

from quackmem.schema.models import TrackedSession, TrackedMessage
from quackmem.schema.enums import MessageStatus

logger = logging.getLogger(__name__)

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


async def _fire_write(
    wrapper: BaseWrapper,
    decorator_kwargs: dict,
    args: tuple,
    fn_kwargs: dict,
    result: Any,
    session_id: uuid.UUID,
    conversation_id: uuid.UUID,
    metadata: dict,
) -> None:
    messages = wrapper.extract_messages(args, fn_kwargs)
    response = wrapper.extract_response(result)
    token_count = wrapper.extract_token_count(result)

    regenerate_message_id = decorator_kwargs.get("regenerate_message_id")

    from quackmem.backend import get_backend
    backend = get_backend()

    session_model = TrackedSession(
        id=session_id,
        conversation_id=conversation_id,
        metadata=metadata,
    )
    await backend.create_session(session_model)

    # Fetch existing messages to deduplicate — callers often pass the full
    # conversation history on every request.
    existing_rows = await backend.get_messages(session_id)

    # Build a set of (role, content) keys already persisted for O(1) lookup.
    # Using a set rather than a positional prefix comparison avoids false
    # deduplication when the caller reorders or inserts messages mid-history.
    existing_key_set: set[tuple[str, object]] = {
        (str(row["role"]), row["content"]) for row in existing_rows
    }
    # Also maintain an ordered map so we can recover the row ID for parent linking.
    existing_key_to_id: dict[tuple[str, object], uuid.UUID] = {
        (str(row["role"]), row["content"]): uuid.UUID(str(row["id"]))
        for row in existing_rows
    }

    # Insert only new input messages (skip already-persisted ones; skip on regeneration).
    last_input_message_id = None
    if not regenerate_message_id:
        for msg in messages:
            msg_role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            key = (msg_role, msg.content)
            if key in existing_key_set:
                # Already persisted — reuse its ID for parent linking.
                last_input_message_id = existing_key_to_id[key]
                continue
            msg_id = uuid.uuid4()
            message_model = TrackedMessage(
                id=msg_id,
                session_id=session_id,
                conversation_id=conversation_id,
                parent_message_id=None,
                role=msg.role,
                content=msg.content,
                token_count=msg.token_count,
                status=MessageStatus.completed,
                metadata=msg.metadata or {},
            )
            await backend.insert_message(message_model)
            # Track in the local set so duplicate messages within the same
            # incoming list are also deduplicated.
            existing_key_set.add(key)
            existing_key_to_id[key] = msg_id
            last_input_message_id = msg_id

    # Read context after the function returned — nested calls may have set a parent.
    ctx = get_tracking_context()
    parent_message_id = ctx.message_id if ctx else None
    # Fallback: if context has no message_id but we have input messages,
    # the response's parent is the last input message
    if parent_message_id is None and last_input_message_id is not None:
        parent_message_id = last_input_message_id

    if regenerate_message_id:
        regen_id = uuid.UUID(str(regenerate_message_id))
        await backend.update_message(
            regen_id,
            response.content,
            regeneration_count=None,
        )
        ctx = get_tracking_context()
        if ctx is not None:
            ctx.message_id = regen_id
            logger.debug(
                "TrackingContext updated (regeneration)",
                extra={
                    "session_id": str(ctx.session_id),
                    "message_id": str(regen_id),
                    "parent_message_id": str(parent_message_id) if parent_message_id else None,
                },
            )
    else:
        message_id = uuid.uuid4()
        message_model = TrackedMessage(
            id=message_id,
            session_id=session_id,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            role=response.role,
            content=response.content,
            token_count=token_count or response.token_count,
            status=MessageStatus.completed,
            metadata=response.metadata or {},
        )
        await backend.insert_message(message_model)

        ctx = get_tracking_context()
        if ctx is not None and message_id is not None:
            ctx.message_id = message_id
            logger.debug(
                "TrackingContext updated (insert)",
                extra={
                    "session_id": str(ctx.session_id),
                    "message_id": str(message_id),
                    "parent_message_id": str(parent_message_id) if parent_message_id else None,
                },
            )


async def _run_tracked_async(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)

    # MetadataValidationError is a developer error — propagate it.
    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)
    try:
        result = await fn(*args, **fn_kwargs)
    except Exception:
        reset_tracking_context(token)
        raise

    try:
        await _fire_write(wrapper, decorator_kwargs, args, fn_kwargs, result, session_id, conversation_id, metadata)
    except Exception:
        logger.error("Tracking write failed", exc_info=True)
    finally:
        # Persist context so get_tracking_context() works after this call returns.
        save_last_tracking_context(ctx)
        reset_tracking_context(token)

    return result


async def _run_tracked_asyncgen(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)

    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)
    chunks = []
    try:
        async for chunk in fn(*args, **fn_kwargs):
            chunks.append(chunk)
            yield chunk
    except Exception:
        reset_tracking_context(token)
        raise

    # Pass the buffered chunk list as "result" — the wrapper's extract_response handles reassembly.
    try:
        await _fire_write(wrapper, decorator_kwargs, args, fn_kwargs, chunks, session_id, conversation_id, metadata)
    except Exception:
        logger.error("Tracking write (stream) failed", exc_info=True)
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
        result = fn(*args, **fn_kwargs)
    except Exception:
        reset_tracking_context(token)
        raise

    try:
        coro = _fire_write(wrapper, decorator_kwargs, args, fn_kwargs, result, session_id, conversation_id, metadata)
        try:
            # If there is already a running event loop (FastAPI, Jupyter, etc.)
            # we must not call asyncio.run() — schedule as a fire-and-forget task.
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # No running loop — safe to call asyncio.run().
            asyncio.run(coro)
    except Exception:
        logger.error("Tracking write failed", exc_info=True)
    finally:
        save_last_tracking_context(ctx)
        reset_tracking_context(token)

    return result


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
