from app.services.crew_runtime import AnswerCrew


def test_answer_crew_fallback():
    result = AnswerCrew().run("What is the refund policy?", ["Refunds within 14 days."])
    assert result.crew == "answer"
    assert len(result.steps) == 3
    assert "refund" in result.final.lower() or result.final
