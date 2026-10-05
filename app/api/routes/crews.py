from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.deps import get_current_user
from app.models.user import User
from app.services.crew_runtime import AnswerCrew
from app.tasks.crew_tasks import run_answer_crew_task

router = APIRouter(prefix="/api/crews", tags=["crews"])


class CrewRunRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    snippets: list[str] = Field(default_factory=list)
    async_run: bool = False


class CrewRunResponse(BaseModel):
    crew: str
    used_llm: bool
    steps: list[dict[str, Any]]
    final: str
    task_id: str | None = None


@router.post("/answer", response_model=CrewRunResponse)
def run_answer_crew(
    body: CrewRunRequest,
    _: Annotated[User, Depends(get_current_user)],
) -> CrewRunResponse:
    if body.async_run:
        task = run_answer_crew_task.delay(body.question, body.snippets)
        return CrewRunResponse(crew="answer", used_llm=False, steps=[], final="queued", task_id=task.id)
    result = AnswerCrew().run(body.question, body.snippets)
    return CrewRunResponse(crew=result.crew, used_llm=result.used_llm, steps=result.steps, final=result.final)
