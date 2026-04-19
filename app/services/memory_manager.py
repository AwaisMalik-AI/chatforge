"""Short-term (sliding window) and long-term (summarization) conversation memory."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chatbot import (
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
    ResponseEvaluation,
)
from app.services.llm_provider import llm_provider

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    _embed_model = None

    def _get_embed_model():
        global _embed_model
        if _embed_model is None:
            _embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return _embed_model
except Exception:  # pragma: no cover

    def _get_embed_model():
        return None


class MemoryManager:
    """Sliding window of recent messages + optional rolling summary in metadata."""

    SUMMARY_KEY = "rolling_summary"
    RAW_WINDOW = 10

    async def get_context(
        self,
        db: AsyncSession,
        conversation_id: int,
        max_messages: int | None = None,
    ) -> list[dict]:
        n = max_messages or settings.MAX_CONVERSATION_HISTORY
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(n)
        )
        rows = list(result.scalars().all())
        rows.reverse()
        out: list[dict] = []
        conv = await db.get(Conversation, conversation_id)
        meta = (conv.extra_metadata or {}) if conv else {}
        summary = meta.get(self.SUMMARY_KEY)
        if summary:
            out.append({"role": "system", "content": f"Earlier conversation summary:\n{summary}"})
        for m in rows:
            out.append({"role": m.role.value, "content": m.content})
        return out

    async def summarize_conversation(self, db: AsyncSession, conversation_id: int) -> str:
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        msgs = result.scalars().all()
        if not msgs:
            return ""
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in msgs[-100:])
        prompt = [
            {
                "role": "system",
                "content": "Summarize the following dialogue in 2-4 sentences. Preserve key facts and user goals.",
            },
            {"role": "user", "content": transcript},
        ]
        resp = await llm_provider.chat(
            prompt,
            model=settings.LLM_MODEL,
            provider=settings.LLM_PROVIDER,
            temperature=0.3,
            max_tokens=512,
            stream=False,
        )
        assert isinstance(resp, dict)
        summary = resp.get("content") or ""
        conv = await db.get(Conversation, conversation_id)
        if conv:
            meta = dict(conv.extra_metadata or {})
            meta[self.SUMMARY_KEY] = summary
            conv.extra_metadata = meta
            await db.flush()
        return summary

    async def apply_sliding_window_strategy(
        self,
        db: AsyncSession,
        conversation_id: int,
        keep_last: int | None = None,
    ) -> None:
        """Keep last N messages raw; summarize older block into rolling_summary."""
        keep = keep_last or self.RAW_WINDOW
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        all_msgs = list(result.scalars().all())
        if len(all_msgs) <= keep + 1:
            return
        older = all_msgs[: -keep]
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in older)
        prompt = [
            {
                "role": "system",
                "content": "Compress older dialogue into a short factual summary for context.",
            },
            {"role": "user", "content": transcript},
        ]
        resp = await llm_provider.chat(
            prompt,
            model=settings.LLM_MODEL,
            provider=settings.LLM_PROVIDER,
            temperature=0.2,
            max_tokens=400,
            stream=False,
        )
        assert isinstance(resp, dict)
        new_part = resp.get("content") or ""
        conv = await db.get(Conversation, conversation_id)
        meta = dict(conv.extra_metadata or {}) if conv else {}
        prev = meta.get(self.SUMMARY_KEY, "")
        meta[self.SUMMARY_KEY] = (prev + "\n" + new_part).strip() if prev else new_part
        if conv:
            conv.extra_metadata = meta
        old_ids = [m.id for m in older]
        if old_ids:
            await db.execute(delete(ResponseEvaluation).where(ResponseEvaluation.message_id.in_(old_ids)))
        for m in older:
            await db.delete(m)
        await db.flush()

    async def get_relevant_memories(
        self,
        db: AsyncSession,
        query: str,
        conversation_id: int,
        top_k: int = 5,
    ) -> list[str]:
        result = await db.execute(
            select(Message).where(
                Message.conversation_id == conversation_id,
                Message.role.in_([MessageRole.user, MessageRole.assistant]),
            )
        )
        msgs = list(result.scalars().all())
        if not msgs:
            return []
        model = _get_embed_model()
        if model is not None and query.strip():
            texts = [m.content for m in msgs]
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            doc_embs = model.encode(texts, normalize_embeddings=True)
            sims = np.dot(doc_embs, q_emb)
            idx = np.argsort(-sims)[:top_k]
            return [texts[i] for i in idx if sims[i] > 0.15]
        q_low = query.lower().split()
        scored: list[tuple[float, str]] = []
        for m in msgs:
            t = m.content.lower()
            score = sum(1 for w in q_low if w in t)
            scored.append((float(score), m.content))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k] if s[0] > 0]

    async def prune_old_conversations(self, db: AsyncSession, days: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        result = await db.execute(
            select(Conversation).where(
                Conversation.status == ConversationStatus.archived,
                Conversation.updated_at < cutoff,
            )
        )
        n = 0
        for c in result.scalars().all():
            mid = select(Message.id).where(Message.conversation_id == c.id)
            await db.execute(delete(ResponseEvaluation).where(ResponseEvaluation.message_id.in_(mid)))
            await db.execute(delete(Message).where(Message.conversation_id == c.id))
            await db.delete(c)
            n += 1
        return n


memory_manager = MemoryManager()
