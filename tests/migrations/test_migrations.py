"""Tests for migration runner.

Unit tests verify the runner module is importable and has the expected functions.
Integration tests (skip_if_no_db) run real upgrade/downgrade against Postgres.
"""
from __future__ import annotations

import os
import pytest

skip_if_no_db = pytest.mark.skipif(
    not os.environ.get("CONVO_TRACKER_TEST_DB_URL"),
    reason="CONVO_TRACKER_TEST_DB_URL not set",
)


class TestMigrationRunnerImport:
    def test_runner_module_importable(self):
        from convo_tracker.migrations import runner
        assert runner is not None

    def test_upgrade_db_is_callable(self):
        from convo_tracker.migrations.runner import upgrade_db
        assert callable(upgrade_db)

    def test_downgrade_db_is_callable(self):
        from convo_tracker.migrations.runner import downgrade_db
        assert callable(downgrade_db)

    def test_upgrade_and_downgrade_in_public_api(self):
        from convo_tracker import upgrade_db, downgrade_db
        assert callable(upgrade_db)
        assert callable(downgrade_db)


@skip_if_no_db
class TestMigrationIntegration:
    """Requires CONVO_TRACKER_TEST_DB_URL to be set."""

    def test_upgrade_then_downgrade(self):
        import os
        db_url = os.environ["CONVO_TRACKER_TEST_DB_URL"]
        os.environ.setdefault("CONVO_TRACKER_DB_URL", db_url)

        from convo_tracker.migrations.runner import upgrade_db, downgrade_db
        upgrade_db()   # Should not raise
        downgrade_db("base")  # Undo all migrations
        upgrade_db()   # Re-apply — must be idempotent

    def test_upgrade_is_idempotent(self):
        import os
        db_url = os.environ["CONVO_TRACKER_TEST_DB_URL"]
        os.environ.setdefault("CONVO_TRACKER_DB_URL", db_url)

        from convo_tracker.migrations.runner import upgrade_db
        upgrade_db()
        upgrade_db()  # Second run should be a no-op
