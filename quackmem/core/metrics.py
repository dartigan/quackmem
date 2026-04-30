"""Lightweight in-process counters for tracking-write observability.

The library doesn't ship a metrics backend — these are simple monotonic
counters you can scrape (Prometheus, statsd, logs) from your host
application. The values reset only on process restart.

Counters:

- ``writes_attempted``: pre/post-write tracking calls that started.
- ``writes_dropped``: tracking calls swallowed by the safe wrappers
  (storage error, validation error, etc.). Non-zero means tracking data
  is being lost; investigate.
- ``writes_retried``: number of retry attempts at the SQLAlchemy layer
  (i.e. transient PG errors that were re-tried).
- ``reservations_unfinalized``: rows reaped by ``reap_orphans``.
- ``drain_timeouts``: pending background tasks that did not finish before
  ``wait_pending_writes`` timed out (each call increments by the count of
  stragglers).
- ``finalize_status_mismatch``: ``finalize_message`` calls where the
  target row was no longer in ``pending`` state (typically a reaper or
  duplicate finalize race). The row is left untouched.

For richer observability, register a callback via
:func:`set_on_tracking_error` to receive the exception object directly.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counters: dict[str, int] = {
    "writes_attempted": 0,
    "writes_dropped": 0,
    "writes_retried": 0,
    "reservations_unfinalized": 0,
    "drain_timeouts": 0,
    "finalize_status_mismatch": 0,
}


def increment(name: str, delta: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + delta


def snapshot() -> dict[str, int]:
    """Return a copy of current counter values."""
    with _lock:
        return dict(_counters)


def reset() -> None:
    """Reset all counters to zero. Test-only — not intended for production."""
    with _lock:
        for k in _counters:
            _counters[k] = 0


_on_tracking_error: Callable[[BaseException, dict], None] | None = None


def set_on_tracking_error(
    callback: Callable[[BaseException, dict], None] | None,
) -> None:
    """Register a callback fired when a tracking write is dropped.

    The callback receives ``(exc, context)`` where ``context`` is a dict
    with at least ``path`` and ``session_id`` keys. Pass ``None`` to clear.

    Use this to wire dropped writes into Sentry / metrics. The callback
    must not raise — exceptions from it are caught and logged.
    """
    global _on_tracking_error
    _on_tracking_error = callback


def fire_on_tracking_error(exc: BaseException, context: dict) -> None:
    cb = _on_tracking_error
    if cb is None:
        return
    try:
        cb(exc, context)
    except Exception:
        logger.exception("on_tracking_error callback raised")
