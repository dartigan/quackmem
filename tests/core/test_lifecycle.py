"""Unit tests for engine lifecycle helpers (dispose_engine, shutdown_tracker)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_engine_state():
    """Make sure each test starts with a cleared engine."""
    from quackmem.db import session as sess_mod
    sess_mod._engine = None
    sess_mod._session_factory = None
    yield
    sess_mod._engine = None
    sess_mod._session_factory = None


@pytest.mark.asyncio
async def test_dispose_engine_is_noop_when_uninitialised():
    from quackmem.db import dispose_engine

    # Must not raise even when init_engine has never been called.
    await dispose_engine()


@pytest.mark.asyncio
async def test_dispose_engine_calls_engine_dispose_and_clears_state():
    from quackmem.db import dispose_engine
    from quackmem.db import session as sess_mod

    fake_engine = AsyncMock()
    fake_engine.dispose = AsyncMock(return_value=None)
    sess_mod._engine = fake_engine
    sess_mod._session_factory = object()  # any non-None sentinel

    await dispose_engine()

    fake_engine.dispose.assert_awaited_once()
    assert sess_mod._engine is None
    assert sess_mod._session_factory is None


@pytest.mark.asyncio
async def test_dispose_engine_can_be_called_twice():
    """Idempotent — second call is a no-op."""
    from quackmem.db import dispose_engine
    from quackmem.db import session as sess_mod

    fake_engine = AsyncMock()
    fake_engine.dispose = AsyncMock(return_value=None)
    sess_mod._engine = fake_engine

    await dispose_engine()
    await dispose_engine()  # must not raise

    fake_engine.dispose.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_tracker_drains_then_disposes():
    """shutdown_tracker must call wait_pending_writes then dispose_engine, in order."""
    import quackmem

    call_order: list[str] = []

    async def fake_wait(timeout=None):
        call_order.append("wait_pending_writes")
        return 0

    async def fake_dispose():
        call_order.append("dispose_engine")

    with patch("quackmem.wait_pending_writes", side_effect=fake_wait), \
         patch("quackmem.db.session.dispose_engine", side_effect=fake_dispose):
        result = await quackmem.shutdown_tracker(drain_timeout=5)

    assert call_order == ["wait_pending_writes", "dispose_engine"]
    assert result == 0


@pytest.mark.asyncio
async def test_shutdown_tracker_returns_pending_count_on_timeout():
    """When wait_pending_writes returns >0 the same value bubbles up."""
    import quackmem

    async def fake_wait(timeout=None):
        return 3  # 3 tasks still in flight

    async def fake_dispose():
        return None

    with patch("quackmem.wait_pending_writes", side_effect=fake_wait), \
         patch("quackmem.db.session.dispose_engine", side_effect=fake_dispose):
        result = await quackmem.shutdown_tracker(drain_timeout=0.01)

    assert result == 3
