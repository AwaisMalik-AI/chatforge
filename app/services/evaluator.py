"""Heuristic + LLM-assisted response evaluation."""

from __future__ import annotations

import re
from typing import Any

from app.services.rag_engine import rag_engine


class ResponseEvaluator:
    async def relevance_check(self, query: str, response: str) -> float:
        if not response.strip():
            return 0.0
        q_terms = set(w for w in re.findall(r"\w+", query.lower()) if len(w) > 2)
        r_low = response.lower()
        if not q_terms:
            return 0.5
        hits = sum(1 for t in q_terms if t in r_low)
        return min(1.0, hits / len(q_terms))

    def groundedness_check(self, response: str, sources: list[str]) -> float:
        return rag_engine.evaluate_groundedness(response, sources)

    def safety_check(self, response: str) -> float:
        blocklist = (
            "kill yourself",
            "bomb recipe",
            "how to hack",
            "credit card number",
        )
        low = response.lower()
        for b in blocklist:
            if b in low:
                return 0.0
        return 0.95

    async def evaluate(
        self,
        message: str,
        response: str,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        sources = sources or []
        rel = await self.relevance_check(message, response)
        ground = self.groundedness_check(response, sources)
        safety = self.safety_check(response)
        helpful = min(1.0, 0.5 * rel + 0.5 * (0.2 + 0.8 * min(1.0, len(response) / 200)))
        composite = (rel + ground + helpful + safety) / 4.0
        return {
            "relevance_score": rel,
            "groundedness_score": ground,
            "helpfulness_score": helpful,
            "safety_score": safety,
            "composite": composite,
            "sources_used": [{"preview": s[:200]} for s in sources[:10]],
            "evaluation_details": {"method": "heuristic_v1"},
        }


response_evaluator = ResponseEvaluator()
