"""Add tool_calls and tool_call_id columns to tracked_messages

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-30 00:00:00.000000

Adds first-class persistence of LLM tool calls and tool results:

* ``tool_calls`` (JSONB, nullable) — populated on ``role=assistant`` rows
  that invoked tools. Holds the full call array verbatim, so multi-tool
  turns survive a round-trip.
* ``tool_call_id`` (TEXT, nullable) — populated on ``role=tool`` rows.
  Identifies which call this row answers.

Indexes:

* GIN on ``tool_calls`` — supports ``WHERE tool_calls @@ ...`` queries
  for analytics ("messages that called tool X").
* Partial btree on ``tool_call_id`` — most rows are NULL, so a partial
  index is far cheaper than a full one.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

GIN_INDEX = "tracked_messages_tool_calls_gin"
PARTIAL_INDEX = "tracked_messages_tool_call_id"


def get_schema() -> str:
    from alembic import context
    return context.config.get_section_option("alembic", "target_schema") or "public"


def upgrade() -> None:
    schema = get_schema()
    op.add_column(
        "tracked_messages",
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True),
        schema=schema,
    )
    op.add_column(
        "tracked_messages",
        sa.Column("tool_call_id", sa.Text(), nullable=True),
        schema=schema,
    )
    op.create_index(
        GIN_INDEX,
        "tracked_messages",
        ["tool_calls"],
        schema=schema,
        postgresql_using="gin",
    )
    op.create_index(
        PARTIAL_INDEX,
        "tracked_messages",
        ["tool_call_id"],
        schema=schema,
        postgresql_where=sa.text("tool_call_id IS NOT NULL"),
    )


def downgrade() -> None:
    schema = get_schema()
    op.drop_index(PARTIAL_INDEX, table_name="tracked_messages", schema=schema)
    op.drop_index(GIN_INDEX, table_name="tracked_messages", schema=schema)
    op.drop_column("tracked_messages", "tool_call_id", schema=schema)
    op.drop_column("tracked_messages", "tool_calls", schema=schema)
