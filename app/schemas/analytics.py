
from pydantic import BaseModel


class UsageAnalytics(BaseModel):
    total_tokens: int
    conversation_count: int
    message_count: int
    period_days: int


class QualityAnalytics(BaseModel):
    avg_relevance: float
    avg_groundedness: float
    avg_helpfulness: float
    avg_safety: float
    composite: float
    sample_size: int


class ModelPerformance(BaseModel):
    model_used: str
    message_count: int
    avg_latency_ms: float | None
    avg_tokens: float | None


class PopularQuery(BaseModel):
    query_preview: str
    count: int


class AnalyticsModelsResponse(BaseModel):
    models: list[ModelPerformance]


class PopularQueriesResponse(BaseModel):
    queries: list[PopularQuery]
