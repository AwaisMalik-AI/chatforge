"""Cheap groundedness check: claims vs retrieved snippets."""

from __future__ import annotations

import re


_SENT = re.compile(r"[^.!?]+[.!?]")


def evaluate_answer(answer: str, snippets: list[str]) -> dict:
    claims = [s.strip() for s in _SENT.findall(answer or "") if len(s.strip()) > 12]
    blob = " ".join(snippets).lower()
    supported = 0
    unsupported: list[str] = []
    for claim in claims[:12]:
        tokens = [t for t in re.findall(r"[a-z0-9]{4,}", claim.lower()) if t not in {"this", "that", "with", "from"}]
        hits = sum(1 for t in tokens if t in blob)
        ok = bool(blob) and hits >= max(1, len(tokens) // 4)
        if ok:
            supported += 1
        else:
            unsupported.append(claim[:180])
    total = max(1, len(claims) or 1)
    score = supported / total if claims else (0.7 if snippets else 0.3)
    return {
        "claims": len(claims),
        "supported": supported,
        "groundedness": round(min(1.0, score), 2),
        "unsupported": unsupported[:5],
        "pass": score >= 0.6,
    }
