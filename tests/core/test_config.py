"""Tests for TrackerConfig Pydantic model."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from quackmem.core.config import TrackerConfig


class TestTrackerConfigValidation:
    """Test field-level validation in TrackerConfig."""

    def test_valid_config_with_asyncpg_url(self):
        """Valid config with postgresql+asyncpg:// URL should work unchanged."""
        config = TrackerConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/db"
        )
        assert config.database_url == "postgresql+asyncpg://user:pass@localhost/db"
        assert config.schema_name == "public"
        assert config.table_prefix == ""
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.echo is False

    def test_postgres_url_auto_corrects_to_asyncpg(self):
        """postgres:// URL should be auto-corrected to postgresql+asyncpg://."""
        config = TrackerConfig(
            database_url="postgres://user:pass@localhost/db"
        )
        assert config.database_url == "postgresql+asyncpg://user:pass@localhost/db"

    def test_postgresql_url_auto_corrects_to_asyncpg(self):
        """postgresql:// URL should be auto-corrected to postgresql+asyncpg://."""
        config = TrackerConfig(
            database_url="postgresql://user:pass@localhost/db"
        )
        assert config.database_url == "postgresql+asyncpg://user:pass@localhost/db"

    def test_database_url_whitespace_stripped(self):
        """Leading/trailing whitespace in database_url should be stripped."""
        config = TrackerConfig(
            database_url="  postgresql+asyncpg://user:pass@localhost/db  "
        )
        assert config.database_url == "postgresql+asyncpg://user:pass@localhost/db"

    def test_empty_database_url_raises_validation_error(self):
        """Empty database_url should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(database_url="")
        assert "database_url must not be empty" in str(exc_info.value)

    def test_whitespace_only_database_url_raises_validation_error(self):
        """Whitespace-only database_url should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(database_url="   ")
        assert "database_url must not be empty" in str(exc_info.value)

    def test_non_postgres_url_raises_validation_error(self):
        """Non-PostgreSQL database URL should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(database_url="mysql://user:pass@localhost/db")
        assert "must be a PostgreSQL URL" in str(exc_info.value)

    def test_empty_schema_name_raises_validation_error(self):
        """Empty schema_name should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(
                database_url="postgresql+asyncpg://user:pass@localhost/db",
                schema_name=""
            )
        assert "schema_name must not be empty" in str(exc_info.value)

    def test_whitespace_only_schema_name_raises_validation_error(self):
        """Whitespace-only schema_name should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(
                database_url="postgresql+asyncpg://user:pass@localhost/db",
                schema_name="   "
            )
        assert "schema_name must not be empty" in str(exc_info.value)

    def test_schema_name_whitespace_stripped(self):
        """Leading/trailing whitespace in schema_name should be stripped."""
        config = TrackerConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="  my_schema  "
        )
        assert config.schema_name == "my_schema"

    def test_pool_size_zero_raises_validation_error(self):
        """pool_size of 0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(
                database_url="postgresql+asyncpg://user:pass@localhost/db",
                pool_size=0
            )
        assert "pool_size must be at least 1" in str(exc_info.value)

    def test_pool_size_negative_raises_validation_error(self):
        """Negative pool_size should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(
                database_url="postgresql+asyncpg://user:pass@localhost/db",
                pool_size=-5
            )
        assert "pool_size must be at least 1" in str(exc_info.value)

    def test_max_overflow_negative_raises_validation_error(self):
        """Negative max_overflow should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(
                database_url="postgresql+asyncpg://user:pass@localhost/db",
                max_overflow=-1
            )
        assert "max_overflow must be non-negative" in str(exc_info.value)

    def test_max_overflow_zero_allowed(self):
        """max_overflow of 0 should be allowed."""
        config = TrackerConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/db",
            max_overflow=0
        )
        assert config.max_overflow == 0

    def test_valid_config_with_all_fields(self):
        """Config with all fields specified should be valid."""
        config = TrackerConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="custom_schema",
            table_prefix="app_",
            pool_size=10,
            max_overflow=20,
            echo=True
        )
        assert config.database_url == "postgresql+asyncpg://user:pass@localhost/db"
        assert config.schema_name == "custom_schema"
        assert config.table_prefix == "app_"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.echo is True


class TestTrackerConfigFrozen:
    """Test that TrackerConfig is frozen (immutable)."""

    def test_frozen_model_raises_on_field_assignment(self):
        """Assigning to a field on a frozen config should raise ValidationError."""
        config = TrackerConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/db"
        )
        with pytest.raises(ValidationError):
            config.schema_name = "new_schema"

    def test_frozen_model_raises_on_new_field_assignment(self):
        """Assigning to a new field on a frozen config should raise ValidationError."""
        config = TrackerConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/db"
        )
        with pytest.raises(ValidationError):
            config.new_field = "value"
