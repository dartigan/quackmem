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
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_task_max_retries: int = 3
    celery_retry_backoff: int = 2
    sync_mode: bool = False
