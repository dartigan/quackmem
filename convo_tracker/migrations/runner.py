from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config


def _make_alembic_config(schema: str = "public", revision_env: str | None = None) -> Config:
    """Build an Alembic Config pointing at the bundled alembic directory.

    Args:
        schema: Target schema name (default: "public")
        revision_env: Optional revision environment variable

    Returns:
        Configured Alembic Config object
    """
    # Build path to alembic.ini relative to this file
    migrations_dir = Path(__file__).parent
    alembic_dir = migrations_dir / "alembic"
    ini_path = alembic_dir / "alembic.ini"

    # Create Config pointing to the ini file
    cfg = Config(str(ini_path))

    # Set script_location to the alembic directory
    cfg.set_main_option("script_location", str(alembic_dir))

    # Set target_schema option
    cfg.set_section_option("alembic", "target_schema", schema)

    # Set version_table to avoid conflicts
    cfg.set_section_option("alembic", "version_table", "convo_tracker_alembic_version")

    return cfg


def upgrade_db(revision: str = "head", schema: str = "public") -> None:
    """Run alembic upgrade to the specified revision.

    Args:
        revision: Target revision (default: "head")
        schema: Target schema name (default: "public")
    """
    cfg = _make_alembic_config(schema=schema)
    command.upgrade(cfg, revision)


def downgrade_db(revision: str = "-1", schema: str = "public") -> None:
    """Run alembic downgrade to the specified revision.

    Args:
        revision: Target revision (default: "-1" for one step back)
        schema: Target schema name (default: "public")
    """
    cfg = _make_alembic_config(schema=schema)
    command.downgrade(cfg, revision)
