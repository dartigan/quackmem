from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quackmem.wrappers.base import BaseWrapper

from quackmem.core.context import TrackingContext, set_tracking_context, get_tracking_context, reset_tracking_context
from quackmem.core.registry import validate_metadata
from quackmem.core.exceptions import MetadataValidationError, TrackerConfigError
from quackmem.schema.models import TrackedSession, TrackedMessage
from quackmem.schema.enums import MessageStatus

logger = logging.getLogger(__name__)

_RESERVED_KEYS = frozenset({"conversation_id"})


def _resolve_ids(decorator_kwargs: dict) -> tuple[uuid.UUID, uuid.UUID]:
    session_id = uuid.uuid4()  # always generated
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

    message_id = uuid.uuid4()

    # Read context after the function returned — nested calls may have set a parent.
    ctx = get_tracking_context()
    parent_message_id = ctx.message_id if ctx else None

    session_model = TrackedSession(
        id=session_id,
        conversation_id=conversation_id,
        metadata=metadata,
    )
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

    from quackmem.backend import get_backend
    backend = get_backend()
    await backend.upsert_session(session_model)
    await backend.insert_message(message_model)


async def _run_tracked_async(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)

    # MetadataValidationError and TrackerConfigError are developer errors — propagate them.
    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)
    try:
        result = await fn(*args, **fn_kwargs)
    finally:
        reset_tracking_context(token)

    try:
        await _fire_write(wrapper, decorator_kwargs, args, fn_kwargs, result, session_id, conversation_id, metadata)
    except Exception:
        logger.error("Tracking write failed", exc_info=True)

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
    finally:
        reset_tracking_context(token)

    # Pass the buffered chunk list as "result" — the wrapper's extract_response handles reassembly.
    try:
        await _fire_write(wrapper, decorator_kwargs, args, fn_kwargs, chunks, session_id, conversation_id, metadata)
    except Exception:
        logger.error("Tracking write (stream) failed", exc_info=True)


def _run_tracked_sync(fn, wrapper, decorator_kwargs, args, fn_kwargs):
    session_id, conversation_id = _resolve_ids(decorator_kwargs)
    metadata = _extract_metadata(decorator_kwargs)

    validate_metadata(metadata)

    ctx = TrackingContext(session_id=session_id, conversation_id=conversation_id)
    token = set_tracking_context(ctx)
    try:
        result = fn(*args, **fn_kwargs)
    finally:
        reset_tracking_context(token)

    try:
        asyncio.run(_fire_write(wrapper, decorator_kwargs, args, fn_kwargs, result, session_id, conversation_id, metadata))
    except Exception:
        logger.error("Tracking write failed", exc_info=True)

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
