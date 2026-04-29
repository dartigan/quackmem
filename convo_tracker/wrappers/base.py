from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from convo_tracker.schema.canonical import CanonicalMessage


class BaseWrapper(ABC):

    @abstractmethod
    def extract_messages(self, args: tuple, kwargs: dict) -> list[CanonicalMessage]:
        """Normalize framework-specific inputs into canonical messages."""

    @abstractmethod
    def extract_response(self, result: Any) -> CanonicalMessage:
        """Normalize the function return value into a single assistant CanonicalMessage."""

    def extract_token_count(self, result: Any) -> int | None:
        """Extract token usage from result. Return None if unavailable."""
        return None

    def extract_streaming_chunks(self, result: Any) -> list | None:
        """For streaming results, return buffered chunks. Default: None."""
        return None
