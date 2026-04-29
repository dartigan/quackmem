from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackerConfig:
    database_url: str
    schema_name: str = "public"
    table_prefix: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
