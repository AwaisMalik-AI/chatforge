"""Action execution records with approval workflow (Jira, email, workflows, etc.)."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ActionType(str, enum.Enum):
    create_ticket = "create_ticket"
    draft_email = "draft_email"
    compare_docs = "compare_docs"
    summarize = "summarize"
    create_task = "create_task"
    initiate_workflow = "initiate_workflow"


class ActionStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    executing = "executing"
    completed = "completed"
    failed = "failed"
    rejected = "rejected"


class ActionExecution(Base):
    __tablename__ = "action_executions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    action_type: Mapped[ActionType] = mapped_column(
        SAEnum(ActionType, name="action_type", native_enum=False, length=32),
        nullable=False,
    )
    input_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[ActionStatus] = mapped_column(
        SAEnum(ActionStatus, name="action_status", native_enum=False, length=16),
        nullable=False,
        default=ActionStatus.pending,
    )
    requires_approval: Mapped[bool] = mapped_column(default=True, nullable=False)
    approved_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    execution_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user = relationship("User", back_populates="action_executions", foreign_keys=[user_id])
    approved_by = relationship("User", foreign_keys=[approved_by_id], viewonly=True)
