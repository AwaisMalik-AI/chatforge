"""Action execution with optional admin approval before side effects."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db, require_admin
from app.models.actions import ActionExecution, ActionStatus
from app.models.user import User, UserRole
from app.schemas.actions import ActionApprove, ActionExecuteRequest, ActionExecutionRead, ActionExecutionSummary
from app.services.action_executor import ActionExecutor
from app.tasks.action_tasks import run_action_task

router = APIRouter(prefix="/api/actions", tags=["actions"])


@router.post("/execute", response_model=ActionExecutionRead)
async def execute_action(
    body: ActionExecuteRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    current: Annotated[User, Depends(get_current_user)],
) -> ActionExecution:
    executor = ActionExecutor(db)
    row = await executor.create_execution(current, body.action_type, body.params)
    await db.flush()
    if row.status == ActionStatus.approved:
        run_action_task.delay(row.id)
        response.status_code = status.HTTP_201_CREATED
    else:
        response.status_code = status.HTTP_202_ACCEPTED
    return row


@router.get("", response_model=list[ActionExecutionSummary])
async def list_actions(
    db: Annotated[AsyncSession, Depends(get_db)],
    current: Annotated[User, Depends(get_current_user)],
    status_filter: ActionStatus | None = Query(None, alias="status"),
    limit: int = Query(50, le=200),
) -> list[ActionExecution]:
    stmt = select(ActionExecution).where(ActionExecution.user_id == current.id)
    if status_filter:
        stmt = stmt.where(ActionExecution.status == status_filter)
    stmt = stmt.order_by(ActionExecution.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{action_id}", response_model=ActionExecutionRead)
async def get_action(
    action_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    current: Annotated[User, Depends(get_current_user)],
) -> ActionExecution:
    result = await db.execute(select(ActionExecution).where(ActionExecution.id == action_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    if row.user_id != current.id and current.role != UserRole.admin:
        raise HTTPException(status_code=404, detail="Not found")
    return row


@router.post("/{action_id}/approve", response_model=ActionExecutionRead)
async def approve_action(
    action_id: int,
    body: ActionApprove,
    db: Annotated[AsyncSession, Depends(get_db)],
    approver: Annotated[User, Depends(require_admin)],
) -> ActionExecution:
    result = await db.execute(select(ActionExecution).where(ActionExecution.id == action_id))
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    if not row.requires_approval:
        raise HTTPException(status_code=400, detail="Action does not require approval")
    if row.status != ActionStatus.pending:
        raise HTTPException(status_code=400, detail="Action is not awaiting approval")

    executor = ActionExecutor(db)
    if not body.approve:
        await executor.mark_rejected(action_id, approver.id, body.reason)
        await db.flush()
        result = await db.execute(select(ActionExecution).where(ActionExecution.id == action_id))
        return result.scalar_one()

    await executor.mark_approved(action_id, approver.id)
    await db.flush()
    run_action_task.delay(action_id)
    result = await db.execute(select(ActionExecution).where(ActionExecution.id == action_id))
    return result.scalar_one()
