from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrackerConfig(BaseSettings):
    """Runtime configuration for quackmem.

    Fields can be supplied directly (``TrackerConfig(database_url=...)``) or
    populated from environment variables. Variables use the ``QUACKMEM_``
    prefix:

    - ``QUACKMEM_DATABASE_URL`` (required when not passed explicitly)
    - ``QUACKMEM_SCHEMA_NAME``
    - ``QUACKMEM_TABLE_PREFIX``
    - ``QUACKMEM_POOL_SIZE``
    - ``QUACKMEM_MAX_OVERFLOW``
    - ``QUACKMEM_ECHO``
    - ``QUACKMEM_DEFAULT_READ_LIMIT``

    Direct kwargs always win over environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="QUACKMEM_",
        frozen=True,
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str
    schema_name: str = "public"
    table_prefix: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    default_read_limit: int = 10

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("database_url must not be empty")
        # Auto-correct to asyncpg driver
        v = v.strip()
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        if not v.startswith("postgresql"):
            raise ValueError(f"database_url must be a PostgreSQL URL, got: {v!r}")
        return v

    @field_validator("schema_name")
    @classmethod
    def validate_schema_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("schema_name must not be empty")
        return v.strip()

    @field_validator("pool_size")
    @classmethod
    def validate_pool_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("pool_size must be at least 1")
        return v

    @field_validator("max_overflow")
    @classmethod
    def validate_max_overflow(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_overflow must be non-negative")
        return v

    @field_validator("default_read_limit")
    @classmethod
    def validate_default_read_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("default_read_limit must be at least 1")
        return v
