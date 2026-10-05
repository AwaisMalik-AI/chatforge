"""RAG-aware answer crew: researcher → writer → critic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings


@dataclass
class CrewResult:
    crew: str
    used_llm: bool
    steps: list[dict[str, Any]] = field(default_factory=list)
    final: str = ""


def _llm(system: str, user: str) -> str | None:
    if not settings.LLM_API_KEY:
        return None
    try:
        url = (settings.LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": settings.LLM_MODEL,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


class AnswerCrew:
    def run(self, question: str, snippets: list[str] | None = None) -> CrewResult:
        evidence = snippets or []
        used_llm = False
        research = _llm(
            "You retrieve and rank evidence. Cite only provided snippets.",
            f"Question: {question}\nSnippets: {evidence}",
        ) or f"Evidence count={len(evidence)}. Key terms from question: {question[:200]}"
        writer = _llm(
            "You write a concise grounded answer. If evidence is thin, say so.",
            f"Question: {question}\nResearch notes: {research}",
        ) or f"Answer draft for '{question[:160]}' using {len(evidence)} snippets. {research[:240]}"
        critic = _llm(
            "You are a critic. Flag hallucination risk and tighten the answer.",
            f"Draft: {writer}\nEvidence: {evidence}",
        ) or f"QA: keep claims tied to {len(evidence)} snippets. Refined: {writer[:400]}"
        if settings.LLM_API_KEY:
            used_llm = True
        steps = [
            {"agent": "researcher", "output": research},
            {"agent": "writer", "output": writer},
            {"agent": "critic", "output": critic},
        ]
        return CrewResult(crew="answer", used_llm=used_llm, steps=steps, final=critic)
