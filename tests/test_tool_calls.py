"""Unit tests for tool-call persistence (no DB).

Covers:
* ``record_tool_result`` — argument coercion, parent lookup, conversation_id
  derivation, metadata merging.
* The decorator's post-write forwarding ``response.tool_calls`` into
  ``MessageFinalization`` so they actually land on the assistant row.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest


# ---------------------------------------------------------------------------
# record_tool_result
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_record_environment():
    """Mock the backend methods that record_tool_result depends on."""
    backend = MagicMock()
    convo_id = uuid4()
    backend.find_assistant_for_tool_call = AsyncMock(return_value=None)
    backend.get_conversation_id = AsyncMock(return_value=convo_id)
    backend.insert_tool_result = AsyncMock(side_effect=lambda **kw: uuid4())

    with patch("quackmem.backend.get_backend", return_value=backend):
        yield backend, convo_id


@pytest.mark.asyncio
async def test_record_tool_result_coerces_string_session_id(mock_record_environment):
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    sid = uuid4()
    await record_tool_result(str(sid), "call_1", "the answer", parent_message_id=uuid4())

    assert backend.insert_tool_result.await_count == 1
    delivered_sid = backend.insert_tool_result.call_args.kwargs["session_id"]
    assert isinstance(delivered_sid, UUID)
    assert delivered_sid == sid


@pytest.mark.asyncio
async def test_record_tool_result_looks_up_parent_when_omitted(mock_record_environment):
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    parent = uuid4()
    backend.find_assistant_for_tool_call = AsyncMock(return_value=parent)

    await record_tool_result(uuid4(), "call_1", "x")

    backend.find_assistant_for_tool_call.assert_awaited_once()
    delivered = backend.insert_tool_result.call_args.kwargs["parent_message_id"]
    assert delivered == parent


@pytest.mark.asyncio
async def test_record_tool_result_nulls_parent_when_lookup_misses(mock_record_environment):
    """A failed parent lookup must NOT block insertion — record with NULL link."""
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    backend.find_assistant_for_tool_call = AsyncMock(return_value=None)

    await record_tool_result(uuid4(), "call_orphan", "x")

    delivered = backend.insert_tool_result.call_args.kwargs["parent_message_id"]
    assert delivered is None


@pytest.mark.asyncio
async def test_record_tool_result_explicit_parent_skips_lookup(mock_record_environment):
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    parent = uuid4()
    await record_tool_result(uuid4(), "call_1", "x", parent_message_id=parent)

    backend.find_assistant_for_tool_call.assert_not_called()
    assert backend.insert_tool_result.call_args.kwargs["parent_message_id"] == parent


@pytest.mark.asyncio
async def test_record_tool_result_derives_conversation_id_from_session(mock_record_environment):
    from quackmem import record_tool_result

    backend, expected_convo = mock_record_environment
    await record_tool_result(uuid4(), "call_1", "x", parent_message_id=uuid4())

    delivered = backend.insert_tool_result.call_args.kwargs["conversation_id"]
    assert delivered == expected_convo


@pytest.mark.asyncio
async def test_record_tool_result_explicit_conversation_id_passed_through(mock_record_environment):
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    convo = uuid4()
    await record_tool_result(
        uuid4(), "call_1", "x",
        parent_message_id=uuid4(), conversation_id=convo,
    )

    assert backend.insert_tool_result.call_args.kwargs["conversation_id"] == convo


@pytest.mark.asyncio
async def test_record_tool_result_name_lands_in_metadata(mock_record_environment):
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    await record_tool_result(
        uuid4(), "call_1", "x",
        parent_message_id=uuid4(),
        name="get_weather",
    )

    meta = backend.insert_tool_result.call_args.kwargs["metadata"]
    assert meta["tool_name"] == "get_weather"


@pytest.mark.asyncio
async def test_record_tool_result_user_metadata_preserved(mock_record_environment):
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    await record_tool_result(
        uuid4(), "call_1", "x",
        parent_message_id=uuid4(),
        name="get_weather",
        metadata={"persona": "researcher"},
    )

    meta = backend.insert_tool_result.call_args.kwargs["metadata"]
    assert meta["persona"] == "researcher"
    assert meta["tool_name"] == "get_weather"


@pytest.mark.asyncio
async def test_record_tool_result_unknown_session_raises(mock_record_environment):
    """When conversation_id is omitted and the session row doesn't exist, fail loudly."""
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    backend.get_conversation_id = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="Unknown session_id"):
        await record_tool_result(uuid4(), "call_1", "x", parent_message_id=uuid4())


@pytest.mark.asyncio
async def test_record_tool_result_returns_inserted_id(mock_record_environment):
    from quackmem import record_tool_result

    backend, _ = mock_record_environment
    expected = uuid4()
    backend.insert_tool_result = AsyncMock(return_value=expected)

    got = await record_tool_result(uuid4(), "call_1", "x", parent_message_id=uuid4())
    assert got == expected


# ---------------------------------------------------------------------------
# Decorator forwards response.tool_calls into MessageFinalization
# ---------------------------------------------------------------------------

class TestDecoratorForwardsResponseToolCalls:
    @pytest.fixture(autouse=True)
    def _reset_registry(self, monkeypatch):
        from quackmem.core import registry as reg_module
        monkeypatch.setattr(reg_module, "_metadata_registry", {})

    @pytest.mark.asyncio
    async def test_dict_response_tool_calls_persisted_via_finalize(self):
        """A dict response with ``tool_calls`` must reach ``finalize_message``."""
        import uuid as _uuid

        from quackmem.core.decorator import track
        from quackmem.wrappers.generic import GenericWrapper

        backend = MagicMock()
        backend.create_session = AsyncMock(return_value=None)
        backend.insert_message = AsyncMock(return_value=None)
        backend.get_messages = AsyncMock(return_value=[])
        backend.reserve_assistant_message = AsyncMock(side_effect=lambda r: r)
        backend.finalize_message = AsyncMock(return_value=None)

        calls = [
            {"id": "c1", "name": "search", "arguments": {"q": "weather"}},
            {"id": "c2", "name": "fetch", "arguments": {"url": "..."}},
        ]
        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(GenericWrapper(), session_id=_uuid.uuid4())
            async def my_persona(messages):
                return {"content": "calling tools", "tool_calls": calls}

            await my_persona([{"role": "user", "content": "what's the weather?"}])

        backend.finalize_message.assert_awaited_once()
        finalization = backend.finalize_message.call_args.args[0]
        assert finalization.tool_calls == calls

    @pytest.mark.asyncio
    async def test_string_response_finalizes_with_no_tool_calls(self):
        import uuid as _uuid

        from quackmem.core.decorator import track
        from quackmem.wrappers.generic import GenericWrapper

        backend = MagicMock()
        backend.create_session = AsyncMock(return_value=None)
        backend.insert_message = AsyncMock(return_value=None)
        backend.get_messages = AsyncMock(return_value=[])
        backend.reserve_assistant_message = AsyncMock(side_effect=lambda r: r)
        backend.finalize_message = AsyncMock(return_value=None)

        with patch("quackmem.backend.get_backend", return_value=backend):
            @track(GenericWrapper(), session_id=_uuid.uuid4())
            async def my_persona(messages):
                return "plain text response"

            await my_persona([{"role": "user", "content": "hi"}])

        finalization = backend.finalize_message.call_args.args[0]
        assert finalization.tool_calls is None
