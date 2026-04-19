from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.models.chatbot import DocumentSourceType, KnowledgeBase
from app.models.user import User
from app.schemas.knowledge import (
    DocumentRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    KnowledgeBaseUpdate,
    KnowledgeStats,
    SearchHit,
)
from app.services.rag_engine import rag_engine
from app.tasks.indexing_tasks import reindex_knowledge_base_task

router = APIRouter(prefix="/api/kb", tags=["knowledge"])


def _kb_read(kb: KnowledgeBase) -> KnowledgeBaseRead:
    return KnowledgeBaseRead(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        owner_id=kb.owner_id,
        chunk_strategy=kb.chunk_strategy.value,
        chunk_size=kb.chunk_size,
        chunk_overlap=kb.chunk_overlap,
        total_documents=kb.total_documents,
        total_chunks=kb.total_chunks,
        embedding_model=kb.embedding_model,
        is_active=kb.is_active,
        created_at=kb.created_at,
    )


@router.post("", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
async def create_kb(
    body: KnowledgeBaseCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    from app.models.chatbot import ChunkStrategy

    kb = KnowledgeBase(
        name=body.name,
        description=body.description,
        owner_id=user.id,
        chunk_strategy=ChunkStrategy(body.chunk_strategy),
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        embedding_model=body.embedding_model,
    )
    db.add(kb)
    await db.flush()
    return _kb_read(kb)


@router.get("", response_model=list[KnowledgeBaseRead])
async def list_kb(
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.owner_id == user.id))
    return [_kb_read(k) for k in result.scalars().all()]


@router.get("/{kb_id}", response_model=KnowledgeBaseRead)
async def get_kb(
    kb_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return _kb_read(kb)


@router.patch("/{kb_id}", response_model=KnowledgeBaseRead)
async def update_kb(
    kb_id: int,
    body: KnowledgeBaseUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    from app.models.chatbot import ChunkStrategy

    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    data = body.model_dump(exclude_unset=True)
    if "chunk_strategy" in data and data["chunk_strategy"] is not None:
        data["chunk_strategy"] = ChunkStrategy(data["chunk_strategy"])
    for k, v in data.items():
        setattr(kb, k, v)
    await db.flush()
    return _kb_read(kb)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(
    kb_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(kb)
    await db.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{kb_id}/documents", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    kb_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    raw = await file.read()
    try:
        doc = await rag_engine.ingest_document(
            db,
            kb_id,
            raw,
            file.filename or "upload",
            DocumentSourceType.file,
            title=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return DocumentRead(
        id=doc.id,
        kb_id=doc.kb_id,
        title=doc.title,
        source_type=doc.source_type.value,
        source_path=doc.source_path,
        status=doc.status.value,
        chunk_count=doc.chunk_count,
        error_message=doc.error_message,
        enterprise_metadata=doc.enterprise_metadata,
        created_at=doc.created_at,
    )


@router.get("/{kb_id}/search", response_model=list[SearchHit])
async def kb_search(
    kb_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    q: Annotated[str, Query(min_length=1, description="Search query")],
    top_k: int = 5,
    threshold: float = 0.0,
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    hits = await rag_engine.search(db, kb_id, q, top_k=top_k, score_threshold=threshold)
    out: list[SearchHit] = []
    for h in hits:
        meta = h.get("metadata") or {}
        out.append(
            SearchHit(
                chunk_id=int(meta.get("chunk_id", 0) or 0),
                document_id=int(meta.get("document_id", 0) or 0),
                content=h.get("content", ""),
                score=float(h.get("score", 0)),
                metadata=meta,
            )
        )
    return out


@router.post("/{kb_id}/reindex")
async def reindex_kb(
    kb_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    reindex_knowledge_base_task.delay(kb_id)
    return {"status": "queued", "kb_id": kb_id}


@router.get("/{kb_id}/stats", response_model=KnowledgeStats)
async def kb_stats(
    kb_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return KnowledgeStats(
        kb_id=kb_id,
        total_documents=kb.total_documents,
        total_chunks=kb.total_chunks,
        embedding_model=kb.embedding_model,
    )
