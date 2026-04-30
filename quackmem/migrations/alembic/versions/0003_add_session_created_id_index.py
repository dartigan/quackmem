"""Add composite index (session_id, created_at, id) to tracked_messages

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-30 00:00:00.000000

Backs ``read_messages``: ``ORDER BY created_at DESC, id DESC LIMIT N``
filtered on ``session_id``. Postgres scans a btree backwards as
efficiently as forwards, so a plain ascending composite is the right
shape — no need to declare DESC columns explicitly.
"""
from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

INDEX_NAME = "tracked_messages_session_created_id"


def get_schema() -> str:
    from alembic import context
    return context.config.get_section_option("alembic", "target_schema") or "public"


def upgrade() -> None:
    schema = get_schema()
    op.create_index(
        INDEX_NAME,
        "tracked_messages",
        ["session_id", "created_at", "id"],
        schema=schema,
    )


def downgrade() -> None:
    schema = get_schema()
    op.drop_index(INDEX_NAME, table_name="tracked_messages", schema=schema)
