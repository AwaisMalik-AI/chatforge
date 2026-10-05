from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.deps import get_current_user
from app.models.user import User
from app.services.hallucination_guard import evaluate_answer

router = APIRouter(prefix="/api/evals", tags=["evals"])


class EvalRequest(BaseModel):
    answer: str = Field(..., min_length=3)
    snippets: list[str] = Field(default_factory=list)


@router.post("/groundedness")
def eval_groundedness(body: EvalRequest, _: Annotated[User, Depends(get_current_user)]) -> dict:
    return {"kind": "groundedness", **evaluate_answer(body.answer, body.snippets)}
