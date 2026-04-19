from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class StreamingChunk(BaseModel):
    """Single SSE event payload for token/delta streaming."""

    event: str = Field(description="e.g. token, tool_call, done, error")
    data: str | dict[str, Any] = Field(description="Text delta or structured payload")
    index: int | None = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: int | None = None
    stream: bool = False
    tools: list[str] | None = None


class ChatResponse(BaseModel):
    conversation_id: int
    message_id: int | None = None
    role: str = "assistant"
    content: str
    model_used: str | None = None
    token_count: int | None = None
    latency_ms: int | None = None
    tool_calls: list[dict[str, Any]] | None = None
    sources: list[dict[str, Any]] | None = None
    evaluation: dict[str, Any] | None = None
    suggested_actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional next steps (e.g. create_ticket) inferred from the user query",
    )


class MessageRead(BaseModel):
    id: int
    role: str
    content: str
    token_count: int | None
    latency_ms: int | None
    model_used: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_results: list[dict[str, Any]] | None
    feedback_score: int | None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("role", mode="before")
    @classmethod
    def _role_str(cls, v):
        return v.value if hasattr(v, "value") else v


class ConversationRead(BaseModel):
    id: int
    chatbot_id: int
    session_id: str
    title: str | None
    message_count: int
    total_tokens_used: int
    status: str
    created_at: datetime
    last_message_at: datetime | None

    model_config = {"from_attributes": True}

    @field_validator("status", mode="before")
    @classmethod
    def _status_str(cls, v):
        return v.value if hasattr(v, "value") else v


class ConversationWithMessages(ConversationRead):
    messages: list[MessageRead] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    message_id: int
    score: int = Field(ge=1, le=5, description="1-5 user rating")
