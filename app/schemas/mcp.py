from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MCPServerCreate(BaseModel):
    name: str
    server_type: str = Field(pattern="^(stdio|http)$")
    command: str | None = None
    url: str | None = None
    is_active: bool = True


class MCPServerUpdate(BaseModel):
    name: str | None = None
    command: str | None = None
    url: str | None = None
    is_active: bool | None = None
    tools_available: list[dict[str, Any]] | None = None


class MCPServerRead(BaseModel):
    id: int
    name: str
    server_type: str
    command: str | None
    url: str | None
    tools_available: list[Any]
    is_active: bool
    health_status: str
    last_health_check: datetime | None
    created_by: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ToolTestRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
