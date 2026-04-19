"""Celery application — broker/result from settings."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "chatforge",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["app.tasks.indexing_tasks", "app.tasks.maintenance_tasks", "app.tasks.action_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
