"""Async indexing jobs (run asyncio inside worker)."""

from __future__ import annotations

import asyncio
import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="index_document_task")
def index_document_task(document_id: int, content_hash: str | None, filename: str | None = None) -> str:
    """Re-index a single document from DB + stored content (placeholder hook for object storage)."""

    async def _go():
        from sqlalchemy import select

        from app.core.database import AsyncSessionLocal
        from app.models.chatbot import KnowledgeChunk, KnowledgeDocument

        async with AsyncSessionLocal() as session:
            doc = await session.get(KnowledgeDocument, document_id)
            if not doc:
                return "missing"
            for ch in (
                await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.document_id == doc.id))
            ).scalars():
                await session.delete(ch)
            doc.status = __import__(
                "app.models.chatbot", fromlist=["DocumentStatus"]
            ).DocumentStatus.pending
            await session.commit()
            logger.info("Document %s queued for re-ingest (hash=%s)", document_id, content_hash)
            return "queued"

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_go())
    finally:
        try:
            loop.close()
        except Exception:
            pass


@celery_app.task(name="reindex_knowledge_base_task")
def reindex_knowledge_base_task(kb_id: int) -> str:
    """Mark all documents in KB for reprocessing; workers would batch re-embed."""

    async def _go():
        from sqlalchemy import select

        from app.core.database import AsyncSessionLocal
        from app.models.chatbot import DocumentStatus, KnowledgeDocument

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(KnowledgeDocument).where(KnowledgeDocument.kb_id == kb_id))
            for d in result.scalars().all():
                d.status = DocumentStatus.pending
            await session.commit()
            return "ok"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_go())
    finally:
        loop.close()
