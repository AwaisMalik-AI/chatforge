from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.actions import ActionStatus, ActionType


class ActionExecuteRequest(BaseModel):
    action_type: ActionType
    params: dict[str, Any] = Field(default_factory=dict)


class ActionExecutionRead(BaseModel):
    id: int
    user_id: int
    action_type: ActionType
    input_params: dict[str, Any]
    output: dict[str, Any] | None
    status: ActionStatus
    requires_approval: bool
    approved_by_id: int | None
    execution_time_ms: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ActionExecutionSummary(BaseModel):
    id: int
    action_type: ActionType
    status: ActionStatus
    requires_approval: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ActionApprove(BaseModel):
    approve: bool = True
    reason: str | None = None
