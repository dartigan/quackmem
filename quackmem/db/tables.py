from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()
tracked_sessions: sa.Table | None = None
tracked_messages: sa.Table | None = None


def build_tables(schema: str | None, prefix: str) -> tuple[sa.Table, sa.Table]:
    global tracked_sessions, tracked_messages

    sessions_name = f"{prefix}tracked_sessions"
    messages_name = f"{prefix}tracked_messages"

    # Guard against double-initialisation (hot-reload, multiple init_tracker calls).
    if tracked_sessions is not None and tracked_messages is not None:
        return tracked_sessions, tracked_messages

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
        sa.Column("tool_calls", JSONB, nullable=True),
        sa.Column("tool_call_id", sa.Text, nullable=True),
        sa.Column("token_count", sa.Integer, nullable=True),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("regeneration_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Index(f"{prefix}tracked_messages_metadata_gin", "metadata", postgresql_using="gin"),
        sa.Index(
            f"{prefix}tracked_messages_tool_calls_gin",
            "tool_calls",
            postgresql_using="gin",
        ),
        sa.Index(
            f"{prefix}tracked_messages_tool_call_id",
            "tool_call_id",
            postgresql_where=sa.text("tool_call_id IS NOT NULL"),
        ),
        # Composite index for read_messages: ORDER BY created_at DESC, id DESC LIMIT N.
        # Postgres scans a btree index backwards as efficiently as forwards,
        # so a plain ascending composite is the right shape here.
        sa.Index(
            f"{prefix}tracked_messages_session_created_id",
            "session_id",
            "created_at",
            "id",
        ),
        schema=schema,
    )

    return tracked_sessions, tracked_messages
