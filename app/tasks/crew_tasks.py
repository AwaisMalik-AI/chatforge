from app.services.crew_runtime import AnswerCrew
from app.tasks.celery_app import celery_app


@celery_app.task(name="chatforge.run_answer_crew")
def run_answer_crew_task(question: str, snippets: list[str] | None = None) -> dict:
    result = AnswerCrew().run(question, snippets or [])
    return {"crew": result.crew, "used_llm": result.used_llm, "steps": result.steps, "final": result.final}
