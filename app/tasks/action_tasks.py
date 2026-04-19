"""Async action execution in Celery worker."""

from __future__ import annotations

import asyncio
import logging

from app.core.database import AsyncSessionLocal
from app.services.action_executor import ActionExecutor
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _run_action(execution_id: int) -> None:
    async with AsyncSessionLocal() as db:
        executor = ActionExecutor(db)
        await executor.run_execution(execution_id)
        await db.commit()


@celery_app.task(name="actions.run_execution", bind=True, max_retries=2)
def run_action_task(self, execution_id: int) -> dict:
    try:
        asyncio.run(_run_action(execution_id))
        return {"ok": True, "execution_id": execution_id}
    except Exception as exc:
        logger.exception("actions.run_execution failed")
        raise self.retry(exc=exc, countdown=10) from exc
