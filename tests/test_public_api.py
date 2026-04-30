"""Unit tests for the top-level quackmem.* public API.

These tests exercise the thin wrappers in ``quackmem/__init__.py`` —
argument coercion, default values, and delegation to the backend — using
a mock backend so they run without Postgres.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest


# ---------------------------------------------------------------------------
# read_messages
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_backend_and_config():
    """Patch get_backend() and get_config() inside quackmem.__init__."""
    backend = MagicMock()
    backend.read_messages = AsyncMock(return_value=[])
    backend.reap_orphans = AsyncMock(return_value=0)

    config = MagicMock()
    config.default_read_limit = 10

    with patch("quackmem.backend.get_backend", return_value=backend), \
         patch("quackmem.db.session.get_config", return_value=config):
        yield backend, config


@pytest.mark.asyncio
async def test_read_messages_coerces_string_session_id_to_uuid(fake_backend_and_config):
    """A non-UUID session_id (e.g. a str) must be parsed into a UUID before delegation."""
    from quackmem import read_messages

    backend, _ = fake_backend_and_config
    sid = uuid4()
    await read_messages(str(sid))

    backend.read_messages.assert_awaited_once()
    delivered_id = backend.read_messages.call_args[0][0]
    assert isinstance(delivered_id, UUID)
    assert delivered_id == sid


@pytest.mark.asyncio
async def test_read_messages_passes_uuid_through_unchanged(fake_backend_and_config):
    from quackmem import read_messages

    backend, _ = fake_backend_and_config
    sid = uuid4()
    await read_messages(sid)

    delivered_id = backend.read_messages.call_args[0][0]
    assert delivered_id is sid


@pytest.mark.asyncio
async def test_read_messages_uses_config_default_limit(fake_backend_and_config):
    """When ``limit`` is None the default from TrackerConfig must be substituted."""
    from quackmem import read_messages

    backend, config = fake_backend_and_config
    config.default_read_limit = 7
    await read_messages(uuid4())

    assert backend.read_messages.call_args.kwargs["limit"] == 7


@pytest.mark.asyncio
async def test_read_messages_explicit_limit_wins(fake_backend_and_config):
    from quackmem import read_messages

    backend, config = fake_backend_and_config
    config.default_read_limit = 7
    await read_messages(uuid4(), limit=2)

    assert backend.read_messages.call_args.kwargs["limit"] == 2


@pytest.mark.asyncio
async def test_read_messages_default_status_is_completed_only(fake_backend_and_config):
    """Default ``statuses`` must filter to {completed} so callers don't see pending/failed rows."""
    from quackmem import read_messages
    from quackmem.schema.enums import MessageStatus

    backend, _ = fake_backend_and_config
    await read_messages(uuid4())

    assert backend.read_messages.call_args.kwargs["statuses"] == {MessageStatus.completed}


@pytest.mark.asyncio
async def test_read_messages_explicit_statuses_passed_through(fake_backend_and_config):
    from quackmem import read_messages
    from quackmem.schema.enums import MessageStatus

    backend, _ = fake_backend_and_config
    statuses = {MessageStatus.pending, MessageStatus.failed}
    await read_messages(uuid4(), statuses=statuses)

    assert backend.read_messages.call_args.kwargs["statuses"] == statuses


@pytest.mark.asyncio
async def test_read_messages_validates_returned_rows_into_models(fake_backend_and_config):
    """Backend rows (dicts) must be converted into TrackedMessage instances before return."""
    from quackmem import read_messages
    from quackmem.schema.enums import MessageRole, MessageStatus
    from quackmem.schema.models import TrackedMessage

    backend, _ = fake_backend_and_config
    sid = uuid4()
    backend.read_messages = AsyncMock(return_value=[
        {
            "id": uuid4(),
            "session_id": sid,
            "conversation_id": uuid4(),
            "role": MessageRole.user,
            "content": "hello",
            "tool_calls": None,
            "token_count": None,
            "status": MessageStatus.completed,
            "regeneration_count": 0,
            "metadata": {},
            "error": None,
        },
    ])
    rows = await read_messages(sid)

    assert len(rows) == 1
    assert isinstance(rows[0], TrackedMessage)
    assert rows[0].content == "hello"


# ---------------------------------------------------------------------------
# reap_orphans
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reap_orphans_default_is_ten_minutes(fake_backend_and_config):
    """When ``older_than`` is omitted the wrapper must default to 10 minutes."""
    from quackmem import reap_orphans

    backend, _ = fake_backend_and_config
    await reap_orphans()

    backend.reap_orphans.assert_awaited_once_with(timedelta(minutes=10))


@pytest.mark.asyncio
async def test_reap_orphans_explicit_value_passed_through(fake_backend_and_config):
    from quackmem import reap_orphans

    backend, _ = fake_backend_and_config
    await reap_orphans(timedelta(seconds=30))

    backend.reap_orphans.assert_awaited_once_with(timedelta(seconds=30))


@pytest.mark.asyncio
async def test_reap_orphans_returns_count_from_backend(fake_backend_and_config):
    from quackmem import reap_orphans

    backend, _ = fake_backend_and_config
    backend.reap_orphans = AsyncMock(return_value=4)

    n = await reap_orphans()
    assert n == 4


# ---------------------------------------------------------------------------
# verify_tracker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_tracker_delegates_to_verify_engine():
    """``verify_tracker`` must be a thin awaiter over ``verify_engine``."""
    import quackmem

    fake_verify = AsyncMock(return_value=None)
    with patch("quackmem.db.session.verify_engine", side_effect=fake_verify):
        await quackmem.verify_tracker()

    fake_verify.assert_awaited_once()


# ---------------------------------------------------------------------------
# init_tracker
# ---------------------------------------------------------------------------

def test_init_tracker_initialises_engine_and_tables(sample_config):
    """init_tracker should call init_engine and build_tables with derived args."""
    from quackmem import init_tracker

    with patch("quackmem.db.session.init_engine") as init_engine, \
         patch("quackmem.db.tables.build_tables") as build_tables:
        init_tracker(sample_config)

    init_engine.assert_called_once_with(sample_config)
    # public schema_name must be passed as None (it's the default search_path)
    build_tables.assert_called_once()
    kwargs = build_tables.call_args.kwargs
    assert kwargs["schema"] is None
    assert kwargs["prefix"] == sample_config.table_prefix


def test_init_tracker_passes_non_public_schema_through():
    from quackmem import init_tracker
    from quackmem.core.config import TrackerConfig

    cfg = TrackerConfig(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        schema_name="custom",
    )
    with patch("quackmem.db.session.init_engine"), \
         patch("quackmem.db.tables.build_tables") as build_tables:
        init_tracker(cfg)

    assert build_tables.call_args.kwargs["schema"] == "custom"
