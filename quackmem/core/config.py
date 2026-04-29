from __future__ import annotations

from pydantic import BaseModel, field_validator, ConfigDict


class TrackerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    database_url: str
    schema_name: str = "public"
    table_prefix: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False

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
