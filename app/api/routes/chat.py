import json
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import get_current_user, get_db
from app.models.chatbot import Chatbot, Conversation, ConversationStatus, Message
from app.models.user import User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationRead,
    ConversationWithMessages,
    FeedbackRequest,
    MessageRead,
)
from app.services.chat_engine import chat_engine


def _conversation_read(c: Conversation) -> ConversationRead:
    return ConversationRead(
        id=c.id,
        chatbot_id=c.chatbot_id,
        session_id=c.session_id,
        title=c.title,
        message_count=c.message_count,
        total_tokens_used=c.total_tokens_used,
        status=c.status.value,
        created_at=c.created_at,
        last_message_at=c.last_message_at,
    )

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=None)
async def post_chat(
    chatbot_id: int,
    body: ChatRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    accept: Annotated[str | None, Header()] = None,
):
    bot = await db.get(Chatbot, chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    wants_sse = accept and "text/event-stream" in accept.lower()
    if body.stream and wants_sse:
        async def gen():
            try:
                stream = await chat_engine.chat(
                    db,
                    chatbot_id,
                    body.message,
                    body.conversation_id,
                    stream=True,
                    user_identifier=user.email,
                    tool_names=body.tools,
                )
                async for chunk in stream:
                    ev = chunk.get("event", "message")
                    data = chunk.get("data")
                    payload = data if isinstance(data, str) else json.dumps(data, default=str)
                    yield f"event: {ev}\ndata: {payload}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        result = await chat_engine.chat(
            db,
            chatbot_id,
            body.message,
            body.conversation_id,
            stream=False,
            user_identifier=user.email,
            tool_names=body.tools,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ChatResponse(**result)


@router.get("/conversations", response_model=list[ConversationRead])
async def list_conversations(
    chatbot_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    bot = await db.get(Chatbot, chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    result = await db.execute(
        select(Conversation)
        .where(Conversation.chatbot_id == chatbot_id, Conversation.status != ConversationStatus.deleted)
        .order_by(Conversation.updated_at.desc())
    )
    return [_conversation_read(c) for c in result.scalars().all()]


@router.get("/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Not found")
    bot = await db.get(Chatbot, conv.chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    msgs = sorted(conv.messages, key=lambda m: m.created_at)
    return ConversationWithMessages(
        id=conv.id,
        chatbot_id=conv.chatbot_id,
        session_id=conv.session_id,
        title=conv.title,
        message_count=conv.message_count,
        total_tokens_used=conv.total_tokens_used,
        status=conv.status.value,
        created_at=conv.created_at,
        last_message_at=conv.last_message_at,
        messages=[MessageRead.model_validate(m) for m in msgs],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Not found")
    bot = await db.get(Chatbot, conv.chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    conv.status = ConversationStatus.deleted
    await db.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/conversations/{conversation_id}/feedback")
async def post_feedback(
    conversation_id: int,
    body: FeedbackRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Not found")
    bot = await db.get(Chatbot, conv.chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    msg = await db.get(Message, body.message_id)
    if not msg or msg.conversation_id != conversation_id:
        raise HTTPException(status_code=404, detail="Message not found")
    msg.feedback_score = body.score
    await db.flush()
    return {"ok": True}
