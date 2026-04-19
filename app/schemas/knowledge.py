from datetime import datetime
from typing import Any

from pydantic import BaseModel


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str | None = None
    chunk_strategy: str = "fixed"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    chunk_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_model: str | None = None
    is_active: bool | None = None


class KnowledgeBaseRead(BaseModel):
    id: int
    name: str
    description: str | None
    owner_id: int
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    total_documents: int
    total_chunks: int
    embedding_model: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentRead(BaseModel):
    id: int
    kb_id: int
    title: str
    source_type: str
    source_path: str | None
    status: str
    chunk_count: int
    error_message: str | None
    enterprise_metadata: dict[str, Any] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    content: str
    score: float
    metadata: dict[str, Any] | None = None


class KnowledgeStats(BaseModel):
    kb_id: int
    total_documents: int
    total_chunks: int
    embedding_model: str
