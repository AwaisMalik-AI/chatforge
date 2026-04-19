"""ChatForge — FastAPI entrypoint with CORS and SSE-capable chat routes."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import actions, analytics, auth, chat, chatbots, knowledge, mcp, prompts
from app.core.config import settings
from app.core.database import init_db

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import app.models.actions  # noqa: F401 — register ORM mappers
    import app.models.chatbot  # noqa: F401
    import app.models.user  # noqa: F401

    await init_db()
    logger.info("Database tables ensured (create_all).")
    yield


app = FastAPI(
    title="ChatForge",
    description=(
        "Production LLM Chatbot Platform with RAG (multi-format), Streaming, Memory, "
        "MCP Tools & Action Agents (Jira, email, workflows) with approval gates"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(chatbots.router)
app.include_router(knowledge.router)
app.include_router(prompts.router)
app.include_router(mcp.router)
app.include_router(analytics.router)
app.include_router(actions.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chatforge"}
