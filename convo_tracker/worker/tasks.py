from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from celery import shared_task

from convo_tracker.schema.models import TrackedSession, TrackedMessage
from convo_tracker.schema.enums import MessageStatus

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=2,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def write_message(self, session_dict: dict, message_dict: dict) -> None:
    """Write a session + message to Postgres. Retries on failure."""
    from convo_tracker.backend import get_backend

    session = TrackedSession.model_validate(session_dict)
    message = TrackedMessage.model_validate(message_dict)
    backend = get_backend()
    try:
        asyncio.run(_write(backend, session, message))
    except Exception as exc:
        logger.error("write_message task failed: %s", exc, exc_info=True)
        try:
            update_message_status.delay(str(message.id), MessageStatus.failed.value)
        except Exception:
            pass
        raise self.retry(exc=exc)


@shared_task(bind=True, max_retries=3, default_retry_delay=2)
def update_message_status(self, message_id: str, status: str) -> None:
    """Update the status of a message row."""
    from convo_tracker.backend import get_backend

    backend = get_backend()
    try:
        asyncio.run(backend.update_status(UUID(message_id), MessageStatus(status)))
    except Exception as exc:
        logger.error("update_message_status task failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)


async def _write(backend, session: TrackedSession, message: TrackedMessage) -> None:
    """Internal write helper. Note: asyncio.run() works in Celery tasks because they
    run in a regular thread (not an async context). If the host app uses uvloop or
    an existing event loop, there may be conflicts — acceptable for v1."""
    await backend.upsert_session(session)
    await backend.insert_message(message)


async def sync_write_message(session: TrackedSession, message: TrackedMessage) -> None:
    """Synchronous write path — used when TrackerConfig.sync_mode=True.
    Writes directly to Postgres without Celery. Awaitable."""
    from convo_tracker.backend import get_backend

    backend = get_backend()
    await _write(backend, session, message)


async def sync_update_status(message_id: UUID, status: MessageStatus) -> None:
    """Sync path equivalent of update_message_status task."""
    from convo_tracker.backend import get_backend

    await get_backend().update_status(message_id, status)
