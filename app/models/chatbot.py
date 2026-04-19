"""Chatbot, conversations, knowledge, prompts, MCP, evaluations."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Float,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ResponseFormat(str, enum.Enum):
    text = "text"
    json = "json"
    markdown = "markdown"


class ConversationStatus(str, enum.Enum):
    active = "active"
    archived = "archived"
    deleted = "deleted"


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"


class ChunkStrategy(str, enum.Enum):
    fixed = "fixed"
    sentence = "sentence"
    paragraph = "paragraph"


class DocumentSourceType(str, enum.Enum):
    file = "file"
    url = "url"
    text = "text"


class DocumentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"


class MCPServerType(str, enum.Enum):
    stdio = "stdio"
    http = "http"


class MCPHealthStatus(str, enum.Enum):
    healthy = "healthy"
    unhealthy = "unhealthy"
    unknown = "unknown"


class Chatbot(Base):
    __tablename__ = "chatbots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    llm_provider: Mapped[str] = mapped_column(String(64), default="openai", nullable=False)
    llm_model: Mapped[str] = mapped_column(String(128), nullable=False)
    temperature: Mapped[float] = mapped_column(Float, default=0.7, nullable=False)
    max_tokens: Mapped[int] = mapped_column(Integer, default=4096, nullable=False)
    knowledge_base_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    tools_enabled: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    personality: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_format: Mapped[ResponseFormat] = mapped_column(
        SAEnum(ResponseFormat, name="response_format", native_enum=False, length=16),
        default=ResponseFormat.text,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    api_key_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    owner = relationship("User", back_populates="chatbots", foreign_keys=[owner_id])
    conversations = relationship("Conversation", back_populates="chatbot")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chatbot_id: Mapped[int] = mapped_column(ForeignKey("chatbots.id"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    user_identifier: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[ConversationStatus] = mapped_column(
        SAEnum(ConversationStatus, name="conv_status", native_enum=False, length=16),
        default=ConversationStatus.active,
        nullable=False,
    )
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    chatbot = relationship("Chatbot", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), nullable=False, index=True
    )
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole, name="msg_role", native_enum=False, length=16),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_calls: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    tool_results: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    feedback_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation = relationship("Conversation", back_populates="messages")
    evaluations = relationship("ResponseEvaluation", back_populates="message")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    chunk_strategy: Mapped[ChunkStrategy] = mapped_column(
        SAEnum(ChunkStrategy, name="chunk_strategy", native_enum=False, length=16),
        default=ChunkStrategy.fixed,
        nullable=False,
    )
    chunk_size: Mapped[int] = mapped_column(Integer, default=500, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=50, nullable=False)
    total_documents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    owner = relationship("User", back_populates="knowledge_bases", foreign_keys=[owner_id])
    documents = relationship("KnowledgeDocument", back_populates="knowledge_base")


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[DocumentSourceType] = mapped_column(
        SAEnum(DocumentSourceType, name="doc_source", native_enum=False, length=16),
        nullable=False,
    )
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus, name="doc_status", native_enum=False, length=16),
        default=DocumentStatus.pending,
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    enterprise_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("KnowledgeChunk", back_populates="document", cascade="all, delete-orphan")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_documents.id"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_vector: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    document = relationship("KnowledgeDocument", back_populates="chunks")


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    variables: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    category: Mapped[str] = mapped_column(String(128), default="general", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MCPServerConfig(Base):
    __tablename__ = "mcp_server_configs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    server_type: Mapped[MCPServerType] = mapped_column(
        SAEnum(MCPServerType, name="mcp_server_type", native_enum=False, length=8),
        nullable=False,
    )
    command: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    tools_available: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    health_status: Mapped[MCPHealthStatus] = mapped_column(
        SAEnum(MCPHealthStatus, name="mcp_health", native_enum=False, length=16),
        default=MCPHealthStatus.unknown,
        nullable=False,
    )
    last_health_check: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ResponseEvaluation(Base):
    __tablename__ = "response_evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=False, index=True)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    groundedness_score: Mapped[float] = mapped_column(Float, nullable=False)
    helpfulness_score: Mapped[float] = mapped_column(Float, nullable=False)
    safety_score: Mapped[float] = mapped_column(Float, nullable=False)
    sources_used: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    evaluation_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    message = relationship("Message", back_populates="evaluations")
