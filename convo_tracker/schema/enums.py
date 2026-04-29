from __future__ import annotations

from enum import Enum


class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class MessageStatus(str, Enum):
    pending = "pending"
    streaming = "streaming"
    completed = "completed"
    failed = "failed"
