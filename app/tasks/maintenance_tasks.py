"""Scheduled maintenance: pruning and summarization."""

from __future__ import annotations

import asyncio
import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="prune_conversations_task")
def prune_conversations_task(days: int = 90) -> int:
    async def _go():
        from app.core.database import AsyncSessionLocal
        from app.services.memory_manager import memory_manager

        async with AsyncSessionLocal() as session:
            n = await memory_manager.prune_old_conversations(session, days)
            await session.commit()
            return n

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_go())
    finally:
        loop.close()


@celery_app.task(name="summarize_long_conversations_task")
def summarize_long_conversations_task(min_messages: int = 30) -> int:
    """Summarize conversations with many messages (sliding window prep)."""

    async def _go():
        from sqlalchemy import func, select

        from app.core.database import AsyncSessionLocal
        from app.models.chatbot import Conversation, Message
        from app.services.memory_manager import memory_manager

        async with AsyncSessionLocal() as session:
            q = select(Conversation.id).join(Message).group_by(Conversation.id).having(
                func.count(Message.id) >= min_messages
            )
            ids = (await session.execute(q)).scalars().all()
            n = 0
            for cid in ids:
                try:
                    await memory_manager.apply_sliding_window_strategy(session, cid)
                    n += 1
                except Exception as e:
                    logger.warning("summarize conv %s: %s", cid, e)
            await session.commit()
            return n

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_go())
    finally:
        loop.close()
