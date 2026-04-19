from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.models.chatbot import Chatbot, Conversation, Message, MessageRole, ResponseEvaluation
from app.models.user import User
from app.schemas.analytics import (
    AnalyticsModelsResponse,
    ModelPerformance,
    PopularQueriesResponse,
    PopularQuery,
    QualityAnalytics,
    UsageAnalytics,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/usage", response_model=UsageAnalytics)
async def usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    days: int = 30,
):
    since = datetime.now(UTC) - timedelta(days=days)
    owner_bots = Chatbot.owner_id == user.id
    msg_count = await db.scalar(
        select(func.count(Message.id))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .join(Chatbot, Conversation.chatbot_id == Chatbot.id)
        .where(owner_bots, Message.created_at >= since)
    )
    conv_count = await db.scalar(
        select(func.count(Conversation.id))
        .join(Chatbot, Conversation.chatbot_id == Chatbot.id)
        .where(owner_bots, Conversation.created_at >= since)
    )
    tokens = await db.scalar(
        select(func.coalesce(func.sum(Message.token_count), 0))
        .select_from(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .join(Chatbot, Conversation.chatbot_id == Chatbot.id)
        .where(owner_bots, Message.created_at >= since, Message.role == MessageRole.assistant)
    )
    return UsageAnalytics(
        total_tokens=int(tokens or 0),
        conversation_count=int(conv_count or 0),
        message_count=int(msg_count or 0),
        period_days=days,
    )


@router.get("/quality", response_model=QualityAnalytics)
async def quality(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    q = (
        select(
            func.avg(ResponseEvaluation.relevance_score),
            func.avg(ResponseEvaluation.groundedness_score),
            func.avg(ResponseEvaluation.helpfulness_score),
            func.avg(ResponseEvaluation.safety_score),
            func.count(ResponseEvaluation.id),
        )
        .join(Message, Message.id == ResponseEvaluation.message_id)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .join(Chatbot, Chatbot.id == Conversation.chatbot_id)
        .where(Chatbot.owner_id == user.id)
    )
    row = (await db.execute(q)).one()
    rel, gr, hel, saf, n = row[0] or 0.0, row[1] or 0.0, row[2] or 0.0, row[3] or 0.0, int(row[4] or 0)
    comp = (rel + gr + hel + saf) / 4.0 if n else 0.0
    return QualityAnalytics(
        avg_relevance=float(rel),
        avg_groundedness=float(gr),
        avg_helpfulness=float(hel),
        avg_safety=float(saf),
        composite=float(comp),
        sample_size=n,
    )


@router.get("/models", response_model=AnalyticsModelsResponse)
async def models_perf(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    q = (
        select(
            Message.model_used,
            func.count(Message.id),
            func.avg(Message.latency_ms),
            func.avg(Message.token_count),
        )
        .join(Conversation, Conversation.id == Message.conversation_id)
        .join(Chatbot, Chatbot.id == Conversation.chatbot_id)
        .where(Chatbot.owner_id == user.id, Message.model_used.isnot(None))
        .group_by(Message.model_used)
    )
    rows = (await db.execute(q)).all()
    models = [
        ModelPerformance(
            model_used=r[0] or "unknown",
            message_count=int(r[1] or 0),
            avg_latency_ms=float(r[2]) if r[2] is not None else None,
            avg_tokens=float(r[3]) if r[3] is not None else None,
        )
        for r in rows
    ]
    return AnalyticsModelsResponse(models=models)


@router.get("/popular-queries", response_model=PopularQueriesResponse)
async def popular_queries(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    limit: int = 20,
):
    q = (
        select(Message.content, func.count(Message.id))
        .join(Conversation, Conversation.id == Message.conversation_id)
        .join(Chatbot, Chatbot.id == Conversation.chatbot_id)
        .where(Chatbot.owner_id == user.id, Message.role == MessageRole.user)
        .group_by(Message.content)
        .order_by(func.count(Message.id).desc())
        .limit(limit)
    )
    rows = (await db.execute(q)).all()
    queries = [PopularQuery(query_preview=(r[0] or "")[:200], count=int(r[1])) for r in rows]
    return PopularQueriesResponse(queries=queries)
