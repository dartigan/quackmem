from __future__ import annotations

from quackmem.db.tables import build_tables, metadata
from quackmem.db.session import (
    dispose_engine,
    get_engine,
    get_session,
    init_engine,
    verify_engine,
)

__all__ = [
    "build_tables",
    "metadata",
    "tracked_sessions",
    "tracked_messages",
    "init_engine",
    "dispose_engine",
    "get_session",
    "get_engine",
    "verify_engine",
]


def __getattr__(name: str):
    """Lazily resolve tracked_sessions / tracked_messages.

    These Table objects are populated by ``build_tables()`` (called from
    ``init_tracker``). Re-exporting them via ``from quackmem.db.tables import …``
    at module load would freeze them to ``None`` for any caller using
    ``from quackmem.db import tracked_sessions``. Defer the lookup so the
    live values are always returned.
    """
    if name in ("tracked_sessions", "tracked_messages"):
        from quackmem.db import tables as _tbl
        return getattr(_tbl, name)
    raise AttributeError(f"module 'quackmem.db' has no attribute {name!r}")
