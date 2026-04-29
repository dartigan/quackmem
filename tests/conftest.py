"""Shared fixtures for quackmem tests."""
from __future__ import annotations

import os
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


# ---------------------------------------------------------------------------
# Mock backend fixture (unit tests — no real DB required)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_backend():
    """Mock PostgresBackend for unit tests that patch quackmem.backend.get_backend."""
    with patch("quackmem.backend.get_backend") as mock_get:
        backend = MagicMock()
        backend.create_session = AsyncMock(return_value=None)
        backend.insert_message = AsyncMock(return_value=None)
        backend.update_status = AsyncMock(return_value=None)
        backend.get_messages = AsyncMock(return_value=[])
        mock_get.return_value = backend
        yield backend


@pytest.fixture
def sample_config():
    from quackmem.core.config import TrackerConfig
    return TrackerConfig(
        database_url="postgresql+asyncpg://test:test@localhost/test",
    )


# ---------------------------------------------------------------------------
# Integration test helpers — require QUACKMEM_TEST_DB_URL env var
# ---------------------------------------------------------------------------

TEST_DB_URL = os.environ.get("QUACKMEM_TEST_DB_URL")
skip_if_no_db = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="QUACKMEM_TEST_DB_URL not set",
)
