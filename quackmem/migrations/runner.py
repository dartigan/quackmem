from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def _make_alembic_config(
    schema: str = "public",
    database_url: str | None = None,
) -> Config:
    """Build an Alembic Config pointing at the bundled alembic directory.

    Args:
        schema: Target schema name (default: "public")
        database_url: Optional database URL. When provided it is written into
            the config so env.py can use it directly, removing the need for the
            QUACKMEM_DB_URL environment variable.

    Returns:
        Configured Alembic Config object
    """
    migrations_dir = Path(__file__).parent
    alembic_dir = migrations_dir / "alembic"
    ini_path = alembic_dir / "alembic.ini"

    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str(alembic_dir))
    cfg.set_section_option("alembic", "target_schema", schema)
    cfg.set_section_option("alembic", "version_table", "quackmem_alembic_version")

    if database_url is not None:
        # Normalise to asyncpg driver — required by the async migration runner.
        url = database_url.strip()
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        cfg.set_section_option("alembic", "sqlalchemy.url", url)

    return cfg


def upgrade_db(
    revision: str = "head",
    schema: str = "public",
    database_url: str | None = None,
) -> None:
    """Run alembic upgrade to the specified revision.

    Args:
        revision: Target revision (default: "head")
        schema: Target schema name (default: "public")
        database_url: PostgreSQL connection URL. When omitted the runner falls
            back to the QUACKMEM_DB_URL environment variable or an already-
            initialised engine (via init_tracker).  Providing it here is the
            recommended approach so callers do not need to set env vars.
    """
    cfg = _make_alembic_config(schema=schema, database_url=database_url)
    command.upgrade(cfg, revision)


def downgrade_db(
    revision: str = "-1",
    schema: str = "public",
    database_url: str | None = None,
) -> None:
    """Run alembic downgrade to the specified revision.

    Args:
        revision: Target revision (default: "-1" for one step back)
        schema: Target schema name (default: "public")
        database_url: PostgreSQL connection URL (see upgrade_db).
    """
    cfg = _make_alembic_config(schema=schema, database_url=database_url)
    command.downgrade(cfg, revision)


def generate_migration(
    message: str,
    schema: str = "public",
    database_url: str | None = None,
) -> None:
    """Autogenerate a new Alembic migration by diffing models against the live DB.

    Args:
        message: Description of the migration
        schema: Target schema name (default: "public")
        database_url: PostgreSQL connection URL (see upgrade_db). Falls back to
            QUACKMEM_DB_URL or an already-initialised engine.
    """
    cfg = _make_alembic_config(schema=schema, database_url=database_url)
    command.revision(cfg, message=message, autogenerate=True)
