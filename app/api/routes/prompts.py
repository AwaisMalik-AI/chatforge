"""CRUD for prompt templates and test rendering with {{variable}} placeholders."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.models.chatbot import PromptTemplate
from app.models.user import User
from app.schemas.prompts import (
    PromptRenderRequest,
    PromptRenderResponse,
    PromptTemplateCreate,
    PromptTemplateRead,
    PromptTemplateUpdate,
)

router = APIRouter(prefix="/api/prompts", tags=["prompts"])

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def _render_template(template: str, variables: dict[str, str]) -> tuple[str, list[str]]:
    unresolved: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key in variables:
            return variables[key]
        unresolved.append(key)
        return m.group(0)

    rendered = _PLACEHOLDER.sub(_sub, template)
    return rendered, unresolved


def _to_read(p: PromptTemplate) -> PromptTemplateRead:
    return PromptTemplateRead.model_validate(p)


@router.post("", response_model=PromptTemplateRead, status_code=status.HTTP_201_CREATED)
async def create_prompt(
    body: PromptTemplateCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    p = PromptTemplate(
        name=body.name,
        description=body.description,
        template=body.template,
        variables=list(body.variables or []),
        category=body.category,
        is_active=body.is_active,
        created_by=user.id,
    )
    db.add(p)
    try:
        await db.flush()
    except IntegrityError as e:
        raise HTTPException(status_code=409, detail="Prompt name already exists") from e
    return _to_read(p)


@router.get("", response_model=list[PromptTemplateRead])
async def list_prompts(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await db.execute(select(PromptTemplate).where(PromptTemplate.created_by == user.id))
    return [_to_read(p) for p in result.scalars().all()]


@router.get("/{prompt_id}", response_model=PromptTemplateRead)
async def get_prompt(
    prompt_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    p = await db.get(PromptTemplate, prompt_id)
    if not p or p.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return _to_read(p)


@router.patch("/{prompt_id}", response_model=PromptTemplateRead)
async def update_prompt(
    prompt_id: int,
    body: PromptTemplateUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    p = await db.get(PromptTemplate, prompt_id)
    if not p or p.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    data = body.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(p, k, v)
    try:
        await db.flush()
    except IntegrityError as e:
        raise HTTPException(status_code=409, detail="Conflict updating prompt") from e
    return _to_read(p)


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(
    prompt_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    p = await db.get(PromptTemplate, prompt_id)
    if not p or p.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(p)
    await db.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{prompt_id}/render", response_model=PromptRenderResponse)
async def render_prompt(
    prompt_id: int,
    body: PromptRenderRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    p = await db.get(PromptTemplate, prompt_id)
    if not p or p.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    rendered, unresolved = _render_template(p.template, body.variables)
    return PromptRenderResponse(rendered=rendered, unresolved_placeholders=sorted(set(unresolved)))
