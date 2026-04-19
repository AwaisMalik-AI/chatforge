"""Action executor: Jira, email drafts, doc compare/summarize, tasks, workflows + approval gate."""

from __future__ import annotations

import base64
import logging
import time
import uuid
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.actions import ActionExecution, ActionStatus, ActionType
from app.models.chatbot import KnowledgeChunk
from app.models.user import User
from app.services.llm_provider import llm_provider

logger = logging.getLogger(__name__)


def action_requires_approval(action_type: ActionType, params: dict[str, Any]) -> bool:
    if params.get("force_approval") is True:
        return True
    if params.get("force_approval") is False:
        return False
    return action_type in (
        ActionType.create_ticket,
        ActionType.draft_email,
        ActionType.initiate_workflow,
    )


class ActionExecutor:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()

    async def create_execution(
        self,
        user: User,
        action_type: ActionType,
        params: dict[str, Any],
    ) -> ActionExecution:
        needs = action_requires_approval(action_type, params)
        row = ActionExecution(
            user_id=user.id,
            action_type=action_type,
            input_params=params,
            output=None,
            status=ActionStatus.pending if needs else ActionStatus.approved,
            requires_approval=needs,
            approved_by_id=None,
            execution_time_ms=None,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def mark_rejected(self, execution_id: int, approver_id: int, reason: str | None) -> None:
        result = await self.db.execute(
            select(ActionExecution).where(ActionExecution.id == execution_id)
        )
        row = result.scalar_one_or_none()
        if not row:
            return
        row.status = ActionStatus.rejected
        row.approved_by_id = approver_id
        row.output = {"reason": reason or "rejected"}

    async def mark_approved(self, execution_id: int, approver_id: int) -> ActionExecution | None:
        result = await self.db.execute(
            select(ActionExecution).where(ActionExecution.id == execution_id)
        )
        row = result.scalar_one_or_none()
        if not row:
            return None
        row.status = ActionStatus.approved
        row.approved_by_id = approver_id
        await self.db.flush()
        return row

    async def run_execution(self, execution_id: int) -> ActionExecution | None:
        result = await self.db.execute(
            select(ActionExecution).where(ActionExecution.id == execution_id)
        )
        row = result.scalar_one_or_none()
        if not row or row.status != ActionStatus.approved:
            return row
        row.status = ActionStatus.executing
        await self.db.flush()

        t0 = time.perf_counter()
        user_result = await self.db.execute(select(User).where(User.id == row.user_id))
        user = user_result.scalar_one_or_none()
        if not user:
            row.status = ActionStatus.failed
            row.output = {"error": "user not found"}
            row.execution_time_ms = int((time.perf_counter() - t0) * 1000)
            return row

        try:
            out = await self.execute(row.action_type, row.input_params or {}, user)
            row.output = out
            row.status = ActionStatus.completed
        except Exception as e:
            logger.exception("Action execution failed id=%s", execution_id)
            row.status = ActionStatus.failed
            row.output = {"error": str(e), "type": type(e).__name__}
        row.execution_time_ms = int((time.perf_counter() - t0) * 1000)
        await self.db.flush()
        return row

    async def execute(
        self, action_type: ActionType, params: dict[str, Any], user: User
    ) -> dict[str, Any]:
        if action_type == ActionType.create_ticket:
            return await self.create_jira_ticket(
                title=params.get("title") or "",
                description=params.get("description") or "",
                project=params.get("project"),
                priority=params.get("priority"),
            )
        if action_type == ActionType.draft_email:
            return await self.draft_email(
                to=params.get("to") or "",
                subject=params.get("subject") or "",
                context=params.get("context") or "",
            )
        if action_type == ActionType.compare_docs:
            return await self.compare_documents(
                doc_id_1=int(params.get("doc_id_1") or params.get("document_id_1") or 0),
                doc_id_2=int(params.get("doc_id_2") or params.get("document_id_2") or 0),
            )
        if action_type == ActionType.summarize:
            return await self.summarize_document(
                doc_id=int(params.get("doc_id") or params.get("document_id") or 0),
            )
        if action_type == ActionType.create_task:
            return self.create_task(
                title=params.get("title") or "",
                assignee=params.get("assignee"),
                due_date=params.get("due_date"),
                user=user,
            )
        if action_type == ActionType.initiate_workflow:
            pl = params.get("payload") if isinstance(params.get("payload"), dict) else {}
            if "workflow_name" in params:
                pl = {**pl, "workflow_name": params["workflow_name"]}
            return await self.initiate_workflow(
                webhook_url=params.get("webhook_url"),
                payload=pl,
            )
        raise ValueError(f"Unknown action: {action_type}")

    async def create_jira_ticket(
        self,
        title: str,
        description: str,
        project: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        title = title or "ChatForge ticket"
        description = description or ""
        project = project or self.settings.JIRA_PROJECT_KEY or "CF"
        priority = priority or "Medium"

        base = self.settings.JIRA_BASE_URL
        email = self.settings.JIRA_EMAIL
        token = self.settings.JIRA_API_TOKEN
        if not base or not token or not email:
            return {
                "simulated": True,
                "message": "Jira credentials not configured; returning draft payload only",
                "draft": {
                    "title": title,
                    "description": description,
                    "project": project,
                    "priority": priority,
                },
            }

        url = f"{base.rstrip('/')}/rest/api/3/issue"
        auth = base64.b64encode(f"{email}:{token}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {
            "fields": {
                "project": {"key": project},
                "summary": title[:255],
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description[:32000]}],
                        }
                    ],
                },
                "issuetype": {"name": "Task"},
                "priority": {"name": priority},
            }
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, headers=headers, json=body)
            if r.status_code >= 400:
                return {"ok": False, "status_code": r.status_code, "body": r.text[:2000]}
            data = r.json()
        return {"ok": True, "key": data.get("key"), "id": data.get("id"), "self": data.get("self")}

    async def draft_email(self, to: str, subject: str, context: str) -> dict[str, Any]:
        prompt = (
            f"Write a professional email body (no subject line) to: {to}\n"
            f"Subject intent: {subject}\nContext from knowledge base / user:\n{context}\n"
        )
        resp = await llm_provider.chat(
            [
                {"role": "system", "content": "You write concise, professional enterprise emails."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.4,
            max_tokens=1024,
        )
        body = (resp.get("content") or "").strip()
        return {"to": to, "subject": subject, "body": body}

    async def _load_document_text(self, document_id: int, max_chars: int = 12000) -> str:
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.document_id == document_id)
            .order_by(KnowledgeChunk.chunk_index)
        )
        result = await self.db.execute(stmt)
        chunks = result.scalars().all()
        parts: list[str] = []
        total = 0
        for c in chunks:
            if total >= max_chars:
                break
            piece = (c.content or "")[: max_chars - total]
            parts.append(piece)
            total += len(piece)
        return "\n".join(parts)

    async def compare_documents(self, doc_id_1: int, doc_id_2: int) -> dict[str, Any]:
        t1 = await self._load_document_text(doc_id_1)
        t2 = await self._load_document_text(doc_id_2)
        prompt = (
            "Compare the following two documents. List key similarities, differences, "
            "and risks if policies conflict. Be specific.\n\n"
            f"--- Document A (id {doc_id_1}) ---\n{t1}\n\n"
            f"--- Document B (id {doc_id_2}) ---\n{t2}"
        )
        resp = await llm_provider.chat(
            [
                {"role": "system", "content": "You are a precise engineering analyst."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.2,
            max_tokens=2048,
        )
        text = (resp.get("content") or "").strip()
        return {"doc_id_1": doc_id_1, "doc_id_2": doc_id_2, "comparison": text}

    async def summarize_document(self, doc_id: int) -> dict[str, Any]:
        text = await self._load_document_text(doc_id)
        prompt = f"Summarize the document for an engineering lead. Use bullet points.\n\n{text}"
        resp = await llm_provider.chat(
            [
                {"role": "system", "content": "You summarize technical documents faithfully."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.2,
            max_tokens=1536,
        )
        summary = (resp.get("content") or "").strip()
        return {"doc_id": doc_id, "summary": summary}

    def create_task(
        self,
        title: str,
        assignee: str | None,
        due_date: Any,
        user: User,
    ) -> dict[str, Any]:
        title = title or "Task"
        assignee = assignee or user.email
        task_id = str(uuid.uuid4())
        return {
            "task_id": task_id,
            "title": title,
            "assignee": assignee,
            "due_date": due_date,
            "created_by": user.email,
            "created_at": datetime.now().isoformat(),
            "status": "open",
        }

    async def initiate_workflow(
        self, webhook_url: str | None, payload: dict[str, Any]
    ) -> dict[str, Any]:
        hook = (webhook_url or "").strip() or (self.settings.WORKFLOW_WEBHOOK_URL or "")
        pl = payload if isinstance(payload, dict) else {}
        wf_name = pl.get("workflow_name") or "default"
        body = {
            "workflow": wf_name,
            "payload": pl,
            "requested_at": datetime.now().isoformat(),
        }
        if hook:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(hook, json=body)
            return {
                "ok": r.is_success,
                "status_code": r.status_code,
                "workflow": wf_name,
            }
        return {
            "simulated": True,
            "workflow": wf_name,
            "message": "Set WORKFLOW_WEBHOOK_URL or pass webhook_url in params",
        }
