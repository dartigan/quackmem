"""Structured logging support for quackmem.

quackmem emits log records with rich ``extra`` fields (``session_id``,
``conversation_id``, ``message_id``, ``error_type``…) so production deployments
can ship structured events to log aggregators.

Two ways to consume them:

1. **Default**: do nothing. Records carry the fields as ``LogRecord``
   attributes; any handler that knows how to read them (e.g.
   ``python-json-logger``, Datadog's library) can pick them up.

2. **Built-in JSON formatter**: call :func:`configure_logging(json=True)` to
   attach a stdlib-only JSON formatter to the ``quackmem`` logger. No extra
   dependencies, intended for "good enough" Kubernetes / journald shipping.

Always opt-in. quackmem never installs a handler unless asked, so it does not
fight the host application's logging setup.
"""
from __future__ import annotations

import json
import logging
from typing import Any

# Standard LogRecord attributes — anything else is treated as a structured field.
_STDLIB_RECORD_FIELDS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
})


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter using only the stdlib.

    Emits one JSON object per record with a fixed envelope plus any
    user-supplied ``extra`` fields. Falls back to ``str()`` for non-JSON-
    serialisable values so logging can't crash the caller.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STDLIB_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    *,
    level: str | int = "INFO",
    json: bool = False,
) -> None:
    """Attach a handler to the ``quackmem`` logger.

    Idempotent — safe to call from both library users and tests. Replaces any
    handler this function previously installed; leaves third-party handlers
    untouched.

    Args:
        level: Standard logging level name or numeric level.
        json: When ``True`` use :class:`JsonFormatter`; otherwise a plain
            ``%(asctime)s %(levelname)s %(name)s %(message)s`` formatter.
    """
    root = logging.getLogger("quackmem")
    # Remove any handler we previously installed so reconfiguration works.
    for handler in list(root.handlers):
        if getattr(handler, "_quackmem_managed", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler._quackmem_managed = True  # type: ignore[attr-defined]
    if json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level)
    # Don't propagate to the root logger — avoids double-logging when the host
    # app has its own handler attached at the root.
    root.propagate = False
