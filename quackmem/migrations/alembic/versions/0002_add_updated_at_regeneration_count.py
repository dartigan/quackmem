"""Add updated_at and regeneration_count to tracked_messages

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-02 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def get_schema() -> str:
    from alembic import context
    return context.config.get_section_option("alembic", "target_schema") or "public"


def upgrade() -> None:
    schema = get_schema()
    op.add_column(
        "tracked_messages",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        schema=schema,
    )
    op.add_column(
        "tracked_messages",
        sa.Column("regeneration_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        schema=schema,
    )


def downgrade() -> None:
    schema = get_schema()
    op.drop_column("tracked_messages", "regeneration_count", schema=schema)
    op.drop_column("tracked_messages", "updated_at", schema=schema)
