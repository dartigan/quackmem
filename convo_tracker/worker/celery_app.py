from __future__ import annotations

from celery import Celery

_celery_app: Celery | None = None


def init_celery(broker_url: str, result_backend: str) -> Celery:
    """Initialize and return the Celery app. Called once by init_tracker."""
    global _celery_app
    app = Celery("convo_tracker", broker=broker_url, backend=result_backend)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
    )
    _celery_app = app
    return app


def get_celery_app() -> Celery:
    """Return the initialized Celery app. Raises if not initialized."""
    if _celery_app is None:
        raise RuntimeError("Celery app not initialized. Call init_tracker() first.")
    return _celery_app
