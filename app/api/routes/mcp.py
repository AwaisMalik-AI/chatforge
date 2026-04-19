from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.models.chatbot import MCPServerConfig, MCPServerType, MCPHealthStatus
from app.models.user import User
from app.schemas.mcp import MCPServerCreate, MCPServerRead, MCPServerUpdate, ToolTestRequest
from app.services.mcp.mcp_client import mcp_client
from app.services.mcp.tool_registry import tool_registry

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


def _read(c: MCPServerConfig) -> MCPServerRead:
    return MCPServerRead(
        id=c.id,
        name=c.name,
        server_type=c.server_type.value,
        command=c.command,
        url=c.url,
        tools_available=list(c.tools_available or []),
        is_active=c.is_active,
        health_status=c.health_status.value,
        last_health_check=c.last_health_check,
        created_by=c.created_by,
        created_at=c.created_at,
    )


@router.post("", response_model=MCPServerRead, status_code=status.HTTP_201_CREATED)
async def create_mcp(
    body: MCPServerCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    cfg = MCPServerConfig(
        name=body.name,
        server_type=MCPServerType(body.server_type),
        command=body.command,
        url=body.url,
        is_active=body.is_active,
        created_by=user.id,
    )
    db.add(cfg)
    await db.flush()
    try:
        await tool_registry.register_server(db, cfg)
    except Exception:
        pass
    return _read(cfg)


@router.get("", response_model=list[MCPServerRead])
async def list_mcp(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await db.execute(select(MCPServerConfig).where(MCPServerConfig.created_by == user.id))
    return [_read(c) for c in result.scalars().all()]


@router.get("/{server_id}", response_model=MCPServerRead)
async def get_mcp(
    server_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    c = await db.get(MCPServerConfig, server_id)
    if not c or c.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return _read(c)


@router.patch("/{server_id}", response_model=MCPServerRead)
async def update_mcp(
    server_id: int,
    body: MCPServerUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    c = await db.get(MCPServerConfig, server_id)
    if not c or c.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    data = body.model_dump(exclude_unset=True)
    if "tools_available" in data and data["tools_available"] is not None:
        c.tools_available = data.pop("tools_available")
    for k, v in data.items():
        setattr(c, k, v)
    await db.flush()
    return _read(c)


@router.delete("/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mcp(
    server_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    c = await db.get(MCPServerConfig, server_id)
    if not c or c.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(c)
    await db.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{server_id}/health")
async def mcp_health(
    server_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    c = await db.get(MCPServerConfig, server_id)
    if not c or c.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    ok = await mcp_client.health_check(c.id, c)
    c.health_status = MCPHealthStatus.healthy if ok else MCPHealthStatus.unhealthy
    c.last_health_check = datetime.now(UTC)
    await db.flush()
    return {"healthy": ok, "status": c.health_status.value}


@router.get("/{server_id}/tools")
async def mcp_tools(
    server_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    c = await db.get(MCPServerConfig, server_id)
    if not c or c.created_by != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    tools = await mcp_client.list_tools(c.id, c)
    c.tools_available = tools
    await db.flush()
    return {"tools": tools}


@router.post("/tools/{tool_name}/test")
async def test_tool(
    tool_name: str,
    body: ToolTestRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    _ = user
    result = await tool_registry.execute_tool(db, tool_name, body.arguments)
    return {"tool": tool_name, "result": result}
