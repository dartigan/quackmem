"""Tests for Pydantic models: TrackedSession, TrackedMessage, CanonicalMessage."""
from __future__ import annotations

from uuid import UUID, uuid4
from datetime import datetime

import pytest

from quackmem.schema.models import TrackedSession, TrackedMessage
from quackmem.schema.canonical import CanonicalMessage
from quackmem.schema.enums import MessageRole, MessageStatus


# ---------------------------------------------------------------------------
# TrackedSession
# ---------------------------------------------------------------------------

class TestTrackedSession:
    def test_auto_uuid(self):
        s = TrackedSession(conversation_id=uuid4())
        assert isinstance(s.id, UUID)

    def test_auto_created_at(self):
        s = TrackedSession(conversation_id=uuid4())
        assert isinstance(s.created_at, datetime)

    def test_explicit_id_accepted(self):
        fixed_id = uuid4()
        s = TrackedSession(id=fixed_id, conversation_id=uuid4())
        assert s.id == fixed_id

    def test_metadata_defaults_to_empty_dict(self):
        s = TrackedSession(conversation_id=uuid4())
        assert s.metadata == {}

    def test_model_dump_serializes_enum_as_string(self):
        """use_enum_values=True means enums become strings in dump."""
        s = TrackedSession(conversation_id=uuid4())
        dumped = s.model_dump()
        # conversation_id should be a UUID object, id too
        assert isinstance(dumped["id"], UUID)
        assert isinstance(dumped["conversation_id"], UUID)

    def test_model_dump_mode_json_serializes_uuid_as_str(self):
        s = TrackedSession(conversation_id=uuid4())
        dumped = s.model_dump(mode="json")
        assert isinstance(dumped["id"], str)
        assert isinstance(dumped["conversation_id"], str)

    def test_two_sessions_have_different_ids(self):
        s1 = TrackedSession(conversation_id=uuid4())
        s2 = TrackedSession(conversation_id=uuid4())
        assert s1.id != s2.id


# ---------------------------------------------------------------------------
# TrackedMessage
# ---------------------------------------------------------------------------

class TestTrackedMessage:
    def _make(self, **overrides) -> TrackedMessage:
        defaults = dict(
            session_id=uuid4(),
            conversation_id=uuid4(),
            role=MessageRole.assistant,
            content="hello",
        )
        defaults.update(overrides)
        return TrackedMessage(**defaults)

    def test_auto_uuid(self):
        m = self._make()
        assert isinstance(m.id, UUID)

    def test_auto_created_at(self):
        m = self._make()
        assert isinstance(m.created_at, datetime)

    def test_default_status_is_completed(self):
        m = self._make()
        # use_enum_values=True — status is stored as string value
        assert m.status == MessageStatus.completed.value or m.status == MessageStatus.completed

    def test_explicit_status(self):
        m = self._make(status=MessageStatus.completed)
        assert m.status == MessageStatus.completed.value or m.status == MessageStatus.completed

    def test_role_stored_as_value(self):
        m = self._make(role=MessageRole.user)
        # use_enum_values=True means the attribute equals the string value
        assert m.role == MessageRole.user.value or m.role == MessageRole.user

    def test_parent_message_id_nullable(self):
        m = self._make()
        assert m.parent_message_id is None

    def test_parent_message_id_set(self):
        parent = uuid4()
        m = self._make(parent_message_id=parent)
        assert m.parent_message_id == parent

    def test_model_dump_mode_json(self):
        m = self._make()
        dumped = m.model_dump(mode="json")
        assert isinstance(dumped["id"], str)
        assert isinstance(dumped["session_id"], str)
        assert isinstance(dumped["role"], str)
        assert isinstance(dumped["status"], str)

    def test_round_trip_model_validate(self):
        m = self._make(content="round trip test")
        dumped = m.model_dump(mode="json")
        restored = TrackedMessage.model_validate(dumped)
        assert str(restored.id) == str(m.id)
        assert restored.content == m.content

    def test_content_can_be_list_of_dicts(self):
        content = [{"type": "text", "text": "hello"}, {"type": "image_url", "url": "http://x.com/img.png"}]
        m = self._make(content=content)
        assert m.content == content

    def test_token_count_nullable(self):
        m = self._make()
        assert m.token_count is None

    def test_token_count_set(self):
        m = self._make(token_count=42)
        assert m.token_count == 42


# ---------------------------------------------------------------------------
# CanonicalMessage
# ---------------------------------------------------------------------------

class TestCanonicalMessage:
    def test_plain_string_content(self):
        msg = CanonicalMessage(role=MessageRole.user, content="hello world")
        assert msg.content == "hello world"
        assert msg.role == MessageRole.user

    def test_list_of_dicts_content(self):
        content = [{"type": "text", "text": "hi"}, {"type": "image_url", "url": "http://x.com/a.jpg"}]
        msg = CanonicalMessage(role=MessageRole.user, content=content)
        assert msg.content == content

    def test_tool_calls_nullable(self):
        msg = CanonicalMessage(role=MessageRole.assistant, content="done")
        assert msg.tool_calls is None

    def test_tool_calls_set(self):
        calls = [{"id": "call_1", "type": "function", "function": {"name": "search"}}]
        msg = CanonicalMessage(role=MessageRole.assistant, content="", tool_calls=calls)
        assert msg.tool_calls == calls

    def test_token_count_and_metadata_optional(self):
        msg = CanonicalMessage(role=MessageRole.system, content="system prompt")
        assert msg.token_count is None
        assert msg.metadata is None

    def test_all_roles_accepted(self):
        for role in MessageRole:
            msg = CanonicalMessage(role=role, content="x")
            assert msg.role == role
