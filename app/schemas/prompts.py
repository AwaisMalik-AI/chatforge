from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PromptTemplateCreate(BaseModel):
    name: str
    description: str | None = None
    template: str
    variables: list[str] = Field(default_factory=list)
    category: str = "general"
    is_active: bool = True


class PromptTemplateUpdate(BaseModel):
    description: str | None = None
    template: str | None = None
    variables: list[str] | None = None
    category: str | None = None
    version: int | None = None
    is_active: bool | None = None


class PromptTemplateRead(BaseModel):
    id: int
    name: str
    description: str | None
    template: str
    variables: list[Any]
    category: str
    version: int
    is_active: bool
    created_by: int
    created_at: datetime

    model_config = {"from_attributes": True}


class PromptRenderRequest(BaseModel):
    variables: dict[str, str] = Field(default_factory=dict)


class PromptRenderResponse(BaseModel):
    rendered: str
    unresolved_placeholders: list[str] = Field(default_factory=list)
