from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_pending: set[asyncio.Task] = set()


def register_pending_task(task: asyncio.Task) -> None:
    """Track a fire-and-forget tracking write so it can be awaited at shutdown.

    The sync code path schedules tracking writes as background tasks on the
    running event loop. Without a registry those tasks may be cancelled when
    the loop shuts down, silently dropping messages. Calling sites pass the
    task here; ``wait_pending_writes()`` drains the registry.
    """
    _pending.add(task)
    task.add_done_callback(_pending.discard)


def pending_count() -> int:
    """Return the number of tracking writes still in flight."""
    return len(_pending)


async def wait_pending_writes(timeout: float | None = None) -> int:
    """Wait for all pending tracking writes to finish.

    Call from your ASGI shutdown handler so background tracking writes scheduled
    by the sync decorator path drain before the event loop closes.

    Example::

        from contextlib import asynccontextmanager
        from quackmem import wait_pending_writes

        @asynccontextmanager
        async def lifespan(app):
            yield
            await wait_pending_writes(timeout=10)

    Args:
        timeout: Optional timeout in seconds. ``None`` waits indefinitely.

    Returns:
        Number of tasks still running when the timeout elapsed (``0`` on
        clean shutdown).
    """
    pending = list(_pending)
    if not pending:
        return 0
    _, still_pending = await asyncio.wait(pending, timeout=timeout)
    if still_pending:
        logger.warning(
            "wait_pending_writes timed out with %d task(s) still running",
            len(still_pending),
            extra={
                "pending_count": len(still_pending),
                "timeout_seconds": timeout,
            },
        )
    return len(still_pending)
