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

    def test_default_read_limit_zero_raises_validation_error(self):
        """default_read_limit of 0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(
                database_url="postgresql+asyncpg://user:pass@localhost/db",
                default_read_limit=0,
            )
        assert "default_read_limit must be at least 1" in str(exc_info.value)

    def test_default_read_limit_negative_raises_validation_error(self):
        """Negative default_read_limit should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig(
                database_url="postgresql+asyncpg://user:pass@localhost/db",
                default_read_limit=-3,
            )
        assert "default_read_limit must be at least 1" in str(exc_info.value)

    def test_default_read_limit_one_allowed(self):
        """default_read_limit of 1 (the minimum) should be accepted."""
        config = TrackerConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/db",
            default_read_limit=1,
        )
        assert config.default_read_limit == 1

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


class TestTrackerConfigEnvVars:
    """Verify QUACKMEM_* env vars populate TrackerConfig fields."""

    def test_database_url_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "QUACKMEM_DATABASE_URL",
            "postgresql+asyncpg://envuser:envpass@envhost/envdb",
        )
        config = TrackerConfig()
        assert config.database_url == "postgresql+asyncpg://envuser:envpass@envhost/envdb"

    def test_all_fields_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("QUACKMEM_DATABASE_URL", "postgresql://u:p@h/d")
        monkeypatch.setenv("QUACKMEM_SCHEMA_NAME", "telemetry")
        monkeypatch.setenv("QUACKMEM_TABLE_PREFIX", "qm_")
        monkeypatch.setenv("QUACKMEM_POOL_SIZE", "20")
        monkeypatch.setenv("QUACKMEM_MAX_OVERFLOW", "30")
        monkeypatch.setenv("QUACKMEM_ECHO", "true")

        config = TrackerConfig()
        assert config.database_url == "postgresql+asyncpg://u:p@h/d"
        assert config.schema_name == "telemetry"
        assert config.table_prefix == "qm_"
        assert config.pool_size == 20
        assert config.max_overflow == 30
        assert config.echo is True

    def test_kwargs_override_env_vars(self, monkeypatch):
        """Direct kwargs win over environment variables."""
        monkeypatch.setenv(
            "QUACKMEM_DATABASE_URL",
            "postgresql+asyncpg://envuser:p@envhost/envdb",
        )
        monkeypatch.setenv("QUACKMEM_POOL_SIZE", "99")
        config = TrackerConfig(
            database_url="postgresql+asyncpg://argval:p@h/d",
            pool_size=3,
        )
        assert "argval" in config.database_url
        assert config.pool_size == 3

    def test_missing_database_url_raises(self, monkeypatch):
        """No env var, no kwarg → ValidationError on the required field."""
        monkeypatch.delenv("QUACKMEM_DATABASE_URL", raising=False)
        with pytest.raises(ValidationError):
            TrackerConfig()

    def test_env_var_validation_runs(self, monkeypatch):
        """Validators (e.g. URL must be Postgres) run on env-loaded values."""
        monkeypatch.setenv("QUACKMEM_DATABASE_URL", "mysql://u:p@h/d")
        with pytest.raises(ValidationError) as exc_info:
            TrackerConfig()
        assert "must be a PostgreSQL URL" in str(exc_info.value)

    def test_env_var_case_insensitive(self, monkeypatch):
        """Lower-case QUACKMEM_database_url also works (case_sensitive=False)."""
        monkeypatch.delenv("QUACKMEM_DATABASE_URL", raising=False)
        monkeypatch.setenv("quackmem_database_url", "postgresql://u:p@h/d")
        config = TrackerConfig()
        assert config.database_url == "postgresql+asyncpg://u:p@h/d"
