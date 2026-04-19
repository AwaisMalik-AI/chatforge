from datetime import datetime

from pydantic import BaseModel, Field


class ChatbotBase(BaseModel):
    name: str
    description: str | None = None
    system_prompt: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    knowledge_base_ids: list[int] = Field(default_factory=list)
    tools_enabled: list[str] = Field(default_factory=list)
    personality: str | None = None
    response_format: str = "text"
    is_active: bool = True


class ChatbotCreate(ChatbotBase):
    pass


class ChatbotUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    knowledge_base_ids: list[int] | None = None
    tools_enabled: list[str] | None = None
    personality: str | None = None
    response_format: str | None = None
    is_active: bool | None = None


class ChatbotRead(ChatbotBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatbotTestRequest(BaseModel):
    message: str = "Hello, respond briefly."
