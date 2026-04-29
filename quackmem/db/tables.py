from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import JSONB

from quackmem.core.config import TrackerConfig

metadata = MetaData()
tracked_sessions: sa.Table | None = None
tracked_messages: sa.Table | None = None


def build_tables(schema: str | None, prefix: str) -> tuple[sa.Table, sa.Table]:
    global tracked_sessions, tracked_messages

    sessions_name = f"{prefix}tracked_sessions"
    messages_name = f"{prefix}tracked_messages"

    if schema:
        self_fk_ref = f"{schema}.{messages_name}.id"
    else:
        self_fk_ref = f"{messages_name}.id"

    tracked_sessions = sa.Table(
        sessions_name,
        metadata,
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", sa.UUID, nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Index(f"{prefix}tracked_sessions_metadata_gin", "metadata", postgresql_using="gin"),
        schema=schema,
    )

    tracked_messages = sa.Table(
        messages_name,
        metadata,
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "session_id",
            sa.UUID,
            sa.ForeignKey(f"{sessions_name}.id" if not schema else f"{schema}.{sessions_name}.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("conversation_id", sa.UUID, nullable=False, index=True),
        sa.Column(
            "parent_message_id",
            sa.UUID,
            sa.ForeignKey(self_fk_ref),
            nullable=True,
            index=True,
        ),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("content", JSONB, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=True),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Index(f"{prefix}tracked_messages_metadata_gin", "metadata", postgresql_using="gin"),
        schema=schema,
    )

    return tracked_sessions, tracked_messages


def _apply_config(config: TrackerConfig) -> None:
    schema = config.schema_name if config.schema_name and config.schema_name != "public" else None
    build_tables(schema=schema, prefix=config.table_prefix)
