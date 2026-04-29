"""Initial schema: tracked_sessions and tracked_messages

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def get_schema() -> str:
    from alembic import context

    cfg = context.config
    return cfg.get_section_option("alembic", "target_schema") or "public"


def upgrade() -> None:
    schema = get_schema()

    op.create_table(
        "tracked_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=schema,
    )
    op.create_index("ix_tracked_sessions_conversation_id", "tracked_sessions", ["conversation_id"], schema=schema)
    op.create_index(
        "ix_tracked_sessions_metadata_gin",
        "tracked_sessions",
        ["metadata"],
        schema=schema,
        postgresql_using="gin",
    )

    op.create_table(
        "tracked_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("parent_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["session_id"],
            [f"{schema}.tracked_sessions.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["parent_message_id"], [f"{schema}.tracked_messages.id"]),
        schema=schema,
    )
    op.create_index("ix_tracked_messages_session_id", "tracked_messages", ["session_id"], schema=schema)
    op.create_index("ix_tracked_messages_conversation_id", "tracked_messages", ["conversation_id"], schema=schema)
    op.create_index(
        "ix_tracked_messages_parent_message_id",
        "tracked_messages",
        ["parent_message_id"],
        schema=schema,
    )
    op.create_index(
        "ix_tracked_messages_metadata_gin",
        "tracked_messages",
        ["metadata"],
        schema=schema,
        postgresql_using="gin",
    )


def downgrade() -> None:
    schema = get_schema()
    op.drop_table("tracked_messages", schema=schema)
    op.drop_table("tracked_sessions", schema=schema)
