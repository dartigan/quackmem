"""End-to-end tests for tool-call persistence against a real Postgres.

Covers:
* Migration 0004 actually creates the columns and indexes.
* The decorator persists ``tool_calls`` on the assistant row.
* ``record_tool_result`` round-trips through ``read_messages`` with the
  correct ``parent_message_id`` linkage.
* Multi-tool turns (one assistant row, N tool rows).
"""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


# ---------------------------------------------------------------------------
# Migration smoke check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_migration_creates_columns_and_indexes(initialized_tracker):
    """After upgrade_db, the new columns and indexes must exist on tracked_messages."""
    engine = create_async_engine(initialized_tracker)
    try:
        async with engine.connect() as conn:
            cols = (await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'tracked_messages' "
                "AND column_name IN ('tool_calls', 'tool_call_id')"
            ))).all()
            col_names = {r[0] for r in cols}
            assert col_names == {"tool_calls", "tool_call_id"}

            idxs = (await conn.execute(text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'tracked_messages' "
                "AND indexname IN ("
                "'tracked_messages_tool_calls_gin', "
                "'tracked_messages_tool_call_id'"
                ")"
            ))).all()
            idx_names = {r[0] for r in idxs}
            assert idx_names == {
                "tracked_messages_tool_calls_gin",
                "tracked_messages_tool_call_id",
            }
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Decorator persists tool_calls on assistant row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decorator_persists_response_tool_calls(initialized_tracker):
    """A persona that returns ``{content, tool_calls}`` lands tool_calls in DB."""
    from quackmem import read_messages
    from quackmem.core.decorator import track
    from quackmem.wrappers.generic import GenericWrapper

    sid = uuid4()
    calls = [
        {"id": "c1", "name": "search_kb", "arguments": {"q": "weather SF"}},
        {"id": "c2", "name": "get_time", "arguments": {"tz": "PST"}},
    ]

    @track(GenericWrapper(), session_id=sid)
    async def support_persona(messages):
        return {"content": "Let me check those for you.", "tool_calls": calls}

    await support_persona([{"role": "user", "content": "weather and time?"}])

    rows = await read_messages(sid, limit=10)
    assistant_rows = [r for r in rows if r.role == "assistant"]
    assert len(assistant_rows) == 1
    persisted = assistant_rows[0].tool_calls
    assert persisted is not None
    assert {c["name"] for c in persisted} == {"search_kb", "get_time"}


# ---------------------------------------------------------------------------
# record_tool_result + read_messages round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_tool_result_round_trips(initialized_tracker):
    """A multi-tool turn: 2 calls → 2 results → read_messages returns linked rows."""
    from quackmem import read_messages, record_tool_result
    from quackmem.core.decorator import track
    from quackmem.wrappers.generic import GenericWrapper

    sid = uuid4()
    calls = [
        {"id": "tc_alpha", "name": "search", "arguments": {"q": "foo"}},
        {"id": "tc_beta", "name": "fetch", "arguments": {"url": "..."}},
    ]

    @track(GenericWrapper(), session_id=sid)
    async def persona(messages):
        return {"content": "calling tools", "tool_calls": calls}

    await persona([{"role": "user", "content": "go"}])

    # User runs the tools and reports back. Note: no parent_message_id passed —
    # quackmem has to look it up via the GIN index.
    alpha_id = await record_tool_result(sid, "tc_alpha", "result-of-search", name="search")
    beta_id = await record_tool_result(sid, "tc_beta", "result-of-fetch", name="fetch")

    assert isinstance(alpha_id, UUID)
    assert isinstance(beta_id, UUID)

    rows = await read_messages(sid, limit=10)

    # 1 user + 1 assistant + 2 tool rows
    assert len(rows) == 4
    by_role: dict[str, list] = {}
    for r in rows:
        by_role.setdefault(r.role, []).append(r)

    assert len(by_role["assistant"]) == 1
    assert len(by_role["tool"]) == 2

    assistant = by_role["assistant"][0]
    tool_rows = by_role["tool"]

    # Each tool row links back to the assistant via parent_message_id and
    # carries the right tool_call_id.
    assert all(t.parent_message_id == assistant.id for t in tool_rows)
    assert {t.tool_call_id for t in tool_rows} == {"tc_alpha", "tc_beta"}

    # Tool name landed in metadata.
    by_call_id = {t.tool_call_id: t for t in tool_rows}
    assert by_call_id["tc_alpha"].metadata["tool_name"] == "search"
    assert by_call_id["tc_beta"].metadata["tool_name"] == "fetch"


@pytest.mark.asyncio
async def test_record_tool_result_with_explicit_parent(initialized_tracker):
    """Explicit parent_message_id skips the lookup and is honoured."""
    from quackmem import read_messages, record_tool_result
    from quackmem.core.decorator import track
    from quackmem.wrappers.generic import GenericWrapper

    sid = uuid4()

    @track(GenericWrapper(), session_id=sid)
    async def persona(messages):
        return {"content": "x", "tool_calls": [{"id": "c1", "name": "t"}]}

    await persona([{"role": "user", "content": "go"}])

    rows = await read_messages(sid, limit=10)
    assistant = next(r for r in rows if r.role == "assistant")

    await record_tool_result(
        sid, "c1", "explicit-parent-result",
        parent_message_id=assistant.id,
    )

    rows = await read_messages(sid, limit=10)
    tool_row = next(r for r in rows if r.role == "tool")
    assert tool_row.parent_message_id == assistant.id


@pytest.mark.asyncio
async def test_record_tool_result_missing_call_inserts_with_null_parent(initialized_tracker):
    """If no assistant row matches the tool_call_id, insert with parent=NULL anyway."""
    from quackmem import read_messages, record_tool_result
    from quackmem.core.decorator import track
    from quackmem.wrappers.generic import GenericWrapper

    sid = uuid4()

    @track(GenericWrapper(), session_id=sid)
    async def persona(messages):
        return "no tool calls here"

    await persona([{"role": "user", "content": "go"}])

    new_id = await record_tool_result(sid, "nonexistent_call", "orphan result")
    assert isinstance(new_id, UUID)

    rows = await read_messages(sid, limit=10)
    tool_rows = [r for r in rows if r.role == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0].tool_call_id == "nonexistent_call"
    assert tool_rows[0].parent_message_id is None


@pytest.mark.asyncio
async def test_record_tool_result_unknown_session_raises(initialized_tracker):
    from quackmem import record_tool_result

    with pytest.raises(ValueError, match="Unknown session_id"):
        await record_tool_result(uuid4(), "c1", "x")


@pytest.mark.asyncio
async def test_find_assistant_picks_most_recent_for_repeated_tool_call_id(initialized_tracker):
    """If a tool_call_id appears in two assistant turns, the lookup picks the latest."""
    from quackmem import record_tool_result
    from quackmem.core.decorator import track
    from quackmem.wrappers.generic import GenericWrapper

    sid = uuid4()

    @track(GenericWrapper(), session_id=sid)
    async def persona(messages):
        return {"content": "x", "tool_calls": [{"id": "shared_id", "name": "t"}]}

    await persona([{"role": "user", "content": "first"}])
    await persona([{"role": "user", "content": "second"}])

    from quackmem import read_messages
    rows = await read_messages(sid, limit=20)
    assistants = [r for r in rows if r.role == "assistant"]
    assert len(assistants) == 2
    second_assistant = assistants[-1]  # chronological order

    await record_tool_result(sid, "shared_id", "answer for the second one")

    rows = await read_messages(sid, limit=20)
    tool_row = next(r for r in rows if r.role == "tool")
    assert tool_row.parent_message_id == second_assistant.id
