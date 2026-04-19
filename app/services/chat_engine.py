"""Orchestrates memory, RAG, tools, LLM, evaluation, persistence."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chatbot import Chatbot, Conversation, ConversationStatus, Message, MessageRole
from app.services.evaluator import response_evaluator
from app.services.llm_provider import llm_provider
from app.services.memory_manager import memory_manager
from app.services.mcp.tool_registry import tool_registry
from app.services.rag_engine import rag_engine

logger = logging.getLogger(__name__)


def infer_suggested_actions(message: str) -> list[dict[str, Any]]:
    """Lightweight intent hints for POST /api/actions/execute (keyword + phrase rules)."""
    q = message.lower()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(action_type: str, label: str, params_hint: dict[str, Any]) -> None:
        if action_type in seen:
            return
        seen.add(action_type)
        out.append({"action_type": action_type, "label": label, "params_hint": params_hint})

    if any(
        x in q
        for x in (
            "jira",
            "create a ticket",
            "create ticket",
            "file a ticket",
            "open an issue",
            "file a bug",
        )
    ):
        push("create_ticket", "Create a Jira ticket", {"title": "", "description": ""})
    if any(x in q for x in ("draft email", "write an email", "send an email", "compose email")):
        push("draft_email", "Draft an email", {"to": "", "subject": "", "context": ""})
    if "compare" in q and ("document" in q or "doc " in q or "docs" in q):
        push("compare_docs", "Compare two knowledge documents", {"doc_id_1": 0, "doc_id_2": 0})
    if any(x in q for x in ("summarize the document", "summarize this doc", "summary of the doc")):
        push("summarize", "Summarize a knowledge document", {"doc_id": 0})
    elif "summarize" in q or "tldr" in q or "tl;dr" in q:
        push("summarize", "Summarize a knowledge document", {"doc_id": 0})
    if any(x in q for x in ("create a task", "create task", "assign a task", "new todo")):
        push("create_task", "Create a tracked task", {"title": "", "assignee": "", "due_date": None})
    if any(x in q for x in ("start workflow", "trigger workflow", "run workflow", "call webhook")):
        push(
            "initiate_workflow",
            "Trigger external workflow / webhook",
            {"webhook_url": "", "payload": {}},
        )
    return out


class ChatEngine:
    async def _get_or_create_conversation(
        self,
        db: AsyncSession,
        chatbot_id: int,
        conversation_id: int | None,
        user_identifier: str | None,
    ) -> Conversation:
        if conversation_id:
            conv = await db.get(Conversation, conversation_id)
            if conv and conv.chatbot_id == chatbot_id:
                return conv
        session_id = uuid.uuid4().hex
        conv = Conversation(
            chatbot_id=chatbot_id,
            session_id=session_id,
            user_identifier=user_identifier,
            status=ConversationStatus.active,
        )
        db.add(conv)
        await db.flush()
        return conv

    def _build_system_prompt(self, bot: Chatbot, rag_context: str | None) -> str:
        parts = [bot.system_prompt or ""]
        if bot.personality:
            parts.append(f"Personality: {bot.personality}")
        if rag_context:
            parts.append(rag_context)
        return "\n\n".join(p for p in parts if p)

    async def _gather_rag(
        self,
        db: AsyncSession,
        bot: Chatbot,
        user_message: str,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        kb_ids = bot.knowledge_base_ids or []
        if not kb_ids:
            return None, []
        all_hits: list[dict[str, Any]] = []
        for kb_id in kb_ids:
            try:
                hits = await rag_engine.search(db, int(kb_id), user_message, top_k=5, score_threshold=0.0)
                all_hits.extend(hits)
            except Exception as e:
                logger.warning("RAG search failed kb=%s: %s", kb_id, e)
        if not all_hits:
            return None, []
        all_hits.sort(key=lambda x: -float(x.get("score", 0)))
        top = all_hits[:8]
        augmented = rag_engine.augment_prompt(user_message, top)
        return augmented, top

    async def _run_tool_loop(
        self,
        db: AsyncSession,
        messages: list[dict[str, Any]],
        bot: Chatbot,
        tools: list[dict[str, Any]] | None,
        max_rounds: int = 5,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        tool_calls_trace: list[dict[str, Any]] = []
        tool_results_trace: list[dict[str, Any]] = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
        for _ in range(max_rounds):
            t0 = time.perf_counter()
            resp = await llm_provider.chat(
                messages,
                model=bot.llm_model,
                provider=bot.llm_provider,
                temperature=bot.temperature,
                max_tokens=bot.max_tokens,
                tools=tools or None,
                stream=False,
            )
            assert isinstance(resp, dict)
            usage = resp.get("usage") or {}
            usage_total["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            usage_total["completion_tokens"] += int(usage.get("completion_tokens", 0))
            content = resp.get("content") or ""
            tcalls = resp.get("tool_calls")
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if not tcalls:
                return content, tool_calls_trace, tool_results_trace, {**usage_total, "latency_ms": latency_ms}
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tcalls,
                }
            )
            for tc in tcalls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    args = {}
                result = await tool_registry.execute_tool(db, name, args)
                tool_calls_trace.append(tc)
                tool_results_trace.append({"name": name, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    }
                )
        return content or "", tool_calls_trace, tool_results_trace, usage_total

    async def chat(
        self,
        db: AsyncSession,
        chatbot_id: int,
        message: str,
        conversation_id: int | None,
        stream: bool,
        user_identifier: str | None = None,
        tool_names: list[str] | None = None,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        bot = await db.get(Chatbot, chatbot_id)
        if not bot or not bot.is_active:
            raise ValueError("Chatbot not found or inactive")
        conv = await self._get_or_create_conversation(db, chatbot_id, conversation_id, user_identifier)

        mem_msgs = await memory_manager.get_context(db, conv.id, settings.MAX_CONVERSATION_HISTORY)
        relevant = await memory_manager.get_relevant_memories(db, message, conv.id, top_k=3)
        if relevant:
            mem_msgs.insert(
                0,
                {
                    "role": "system",
                    "content": "Possibly relevant prior turns:\n" + "\n---\n".join(relevant),
                },
            )

        rag_augmented, rag_hits = await self._gather_rag(db, bot, message)
        system_content = self._build_system_prompt(bot, rag_augmented)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        for m in mem_msgs:
            if m["role"] == "system" and m["content"] == system_content:
                continue
            messages.append(m)
        messages.append({"role": "user", "content": message})

        tools = await tool_registry.get_tools_for_chatbot(db, chatbot_id)
        if tool_names:
            allowed = {str(n) for n in tool_names}
            tools = [
                t
                for t in (tools or [])
                if (t.get("function") or {}).get("name") in allowed
            ]

        if stream and settings.STREAMING_ENABLED:
            return self._stream_response(db, bot, conv, message, messages, rag_hits)

        t0 = time.perf_counter()
        content, tcalls, tresults, meta = await self._run_tool_loop(db, messages, bot, tools or None)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        sources = [h.get("content", "") for h in rag_hits]
        eval_result = await response_evaluator.evaluate(message, content, sources)

        um = Message(
            conversation_id=conv.id,
            role=MessageRole.user,
            content=message,
            token_count=llm_provider.count_tokens(message, bot.llm_model),
        )
        db.add(um)
        await db.flush()

        am = Message(
            conversation_id=conv.id,
            role=MessageRole.assistant,
            content=content,
            token_count=llm_provider.count_tokens(content, bot.llm_model),
            latency_ms=latency_ms,
            model_used=bot.llm_model,
            tool_calls=tcalls or None,
            tool_results=tresults or None,
        )
        db.add(am)
        await db.flush()

        from app.models.chatbot import ResponseEvaluation

        ev = ResponseEvaluation(
            message_id=am.id,
            relevance_score=eval_result["relevance_score"],
            groundedness_score=eval_result["groundedness_score"],
            helpfulness_score=eval_result["helpfulness_score"],
            safety_score=eval_result["safety_score"],
            sources_used=eval_result.get("sources_used"),
            evaluation_details=eval_result.get("evaluation_details"),
        )
        db.add(ev)

        conv.message_count = (conv.message_count or 0) + 2
        conv.total_tokens_used = (conv.total_tokens_used or 0) + (am.token_count or 0) + (um.token_count or 0)
        conv.last_message_at = am.created_at
        await db.flush()

        suggested = infer_suggested_actions(message)
        return {
            "conversation_id": conv.id,
            "message_id": am.id,
            "role": "assistant",
            "content": content,
            "model_used": bot.llm_model,
            "token_count": am.token_count,
            "latency_ms": latency_ms,
            "tool_calls": tcalls or None,
            "sources": rag_hits,
            "evaluation": eval_result,
            "suggested_actions": suggested,
        }

    async def _stream_response(
        self,
        db: AsyncSession,
        bot: Chatbot,
        conv: Conversation,
        user_message: str,
        messages: list[dict[str, Any]],
        rag_hits: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream tokens; tool calling is disabled in stream mode for simplicity."""
        tools = None
        um = Message(
            conversation_id=conv.id,
            role=MessageRole.user,
            content=user_message,
            token_count=llm_provider.count_tokens(user_message, bot.llm_model),
        )
        db.add(um)
        await db.flush()
        yield {"event": "start", "data": {"conversation_id": conv.id, "user_message_id": um.id}}

        gen = await llm_provider.chat(
            messages,
            model=bot.llm_model,
            provider=bot.llm_provider,
            temperature=bot.temperature,
            max_tokens=bot.max_tokens,
            tools=tools,
            stream=True,
        )
        buf: list[str] = []
        t0 = time.perf_counter()
        async for chunk in gen:
            buf.append(chunk)
            yield {"event": "token", "data": chunk}
        content = "".join(buf)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        sources = [h.get("content", "") for h in rag_hits]
        eval_result = await response_evaluator.evaluate(user_message, content, sources)

        am = Message(
            conversation_id=conv.id,
            role=MessageRole.assistant,
            content=content,
            token_count=llm_provider.count_tokens(content, bot.llm_model),
            latency_ms=latency_ms,
            model_used=bot.llm_model,
        )
        db.add(am)
        await db.flush()

        from app.models.chatbot import ResponseEvaluation

        db.add(
            ResponseEvaluation(
                message_id=am.id,
                relevance_score=eval_result["relevance_score"],
                groundedness_score=eval_result["groundedness_score"],
                helpfulness_score=eval_result["helpfulness_score"],
                safety_score=eval_result["safety_score"],
                sources_used=eval_result.get("sources_used"),
                evaluation_details=eval_result.get("evaluation_details"),
            )
        )
        conv.message_count = (conv.message_count or 0) + 2
        conv.total_tokens_used = (conv.total_tokens_used or 0) + (am.token_count or 0) + (um.token_count or 0)
        conv.last_message_at = am.created_at
        await db.flush()

        yield {
            "event": "done",
            "data": {
                "message_id": am.id,
                "evaluation": eval_result,
                "sources": rag_hits,
                "suggested_actions": infer_suggested_actions(user_message),
            },
        }


chat_engine = ChatEngine()
