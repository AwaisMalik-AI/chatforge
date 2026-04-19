from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.models.chatbot import Chatbot, ResponseFormat
from app.models.user import User
from app.schemas.chatbot import ChatbotCreate, ChatbotRead, ChatbotTestRequest, ChatbotUpdate
from app.services.chat_engine import chat_engine

router = APIRouter(prefix="/api/chatbots", tags=["chatbots"])


def _to_read(bot: Chatbot) -> ChatbotRead:
    return ChatbotRead(
        id=bot.id,
        name=bot.name,
        description=bot.description,
        system_prompt=bot.system_prompt,
        llm_provider=bot.llm_provider,
        llm_model=bot.llm_model,
        temperature=bot.temperature,
        max_tokens=bot.max_tokens,
        knowledge_base_ids=list(bot.knowledge_base_ids or []),
        tools_enabled=list(bot.tools_enabled or []),
        personality=bot.personality,
        response_format=bot.response_format.value,
        is_active=bot.is_active,
        owner_id=bot.owner_id,
        created_at=bot.created_at,
        updated_at=bot.updated_at,
    )


@router.post("", response_model=ChatbotRead, status_code=status.HTTP_201_CREATED)
async def create_chatbot(
    body: ChatbotCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    bot = Chatbot(
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        llm_provider=body.llm_provider,
        llm_model=body.llm_model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        knowledge_base_ids=body.knowledge_base_ids,
        tools_enabled=body.tools_enabled,
        personality=body.personality,
        response_format=ResponseFormat(body.response_format),
        is_active=body.is_active,
        owner_id=user.id,
    )
    db.add(bot)
    await db.flush()
    return _to_read(bot)


@router.get("", response_model=list[ChatbotRead])
async def list_chatbots(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await db.execute(select(Chatbot).where(Chatbot.owner_id == user.id))
    return [_to_read(b) for b in result.scalars().all()]


@router.get("/{chatbot_id}", response_model=ChatbotRead)
async def get_chatbot(
    chatbot_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    bot = await db.get(Chatbot, chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return _to_read(bot)


@router.patch("/{chatbot_id}", response_model=ChatbotRead)
async def update_chatbot(
    chatbot_id: int,
    body: ChatbotUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    bot = await db.get(Chatbot, chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    data = body.model_dump(exclude_unset=True)
    if "response_format" in data and data["response_format"] is not None:
        data["response_format"] = ResponseFormat(data["response_format"])
    for k, v in data.items():
        setattr(bot, k, v)
    await db.flush()
    return _to_read(bot)


@router.delete("/{chatbot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chatbot(
    chatbot_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    bot = await db.get(Chatbot, chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(bot)
    await db.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{chatbot_id}/test")
async def test_chatbot(
    chatbot_id: int,
    body: ChatbotTestRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    bot = await db.get(Chatbot, chatbot_id)
    if not bot or bot.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    try:
        out = await chat_engine.chat(
            db,
            chatbot_id,
            body.message,
            conversation_id=None,
            stream=False,
            user_identifier=user.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return out
