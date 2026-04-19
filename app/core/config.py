"""Application settings — all secrets from environment (no hardcoding)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database & cache
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://chatforge:chatforge@localhost:5432/chatforge",
        description="Async SQLAlchemy URL",
    )
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Security
    SECRET_KEY: str = Field(default="change-me-in-production-use-openssl-rand")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # LLM
    LLM_PROVIDER: str = Field(default="openai", pattern="^(openai|anthropic|ollama)$")
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str | None = None
    LLM_BASE_URL: str | None = None  # e.g. Ollama http://localhost:11434/v1

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DB_URL: str = Field(
        default="./data/chroma",
        description="Chroma persistent path or HTTP host URL",
    )

    # Celery
    CELERY_BROKER_URL: str | None = None
    CELERY_RESULT_BACKEND: str | None = None

    # Chat / generation
    MAX_CONVERSATION_HISTORY: int = 50
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7
    STREAMING_ENABLED: bool = True

    # MCP: JSON array of {"name","server_type","command","args","url","env"}
    MCP_SERVERS: str = Field(default="[]")

    # Actions: Jira / workflows (optional)
    JIRA_BASE_URL: str | None = None
    JIRA_API_TOKEN: str | None = None
    JIRA_EMAIL: str | None = Field(
        default=None,
        description="Jira Cloud: email for the API token; used with Basic auth",
    )
    JIRA_PROJECT_KEY: str | None = Field(default=None, description="Default Jira project key")
    WORKFLOW_WEBHOOK_URL: str | None = None

    LOG_LEVEL: str = "INFO"

    @field_validator("MCP_SERVERS", mode="before")
    @classmethod
    def _coerce_mcp_servers(cls, v: Any) -> str:
        if v is None:
            return "[]"
        if isinstance(v, list | dict):
            return json.dumps(v)
        return str(v)

    def mcp_servers_list(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.MCP_SERVERS or "[]")
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    @property
    def celery_broker(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
