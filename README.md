# ChatForge — Production LLM Chatbot Platform with RAG, Streaming, MCP & Action Agents

ChatForge is a **backend-first** platform for multi-tenant AI assistants: **retrieval-augmented generation** with **multi-format document ingestion**, **Server-Sent Events (SSE) streaming**, **conversation memory**, **MCP tool servers** (stdio + HTTP), **response evaluation**, **analytics**, and **governed action execution** (Jira, email drafts, document intelligence, tasks, webhooks) with **admin approval gates** for sensitive operations.

---

## ASCII architecture

```
                                    +------------------+
                                    |  User / Client   |
                                    +--------+---------+
                                             |
                         JSON  or  SSE (text/event-stream)
                                             |
                                             v
+--------------------------------------------------------------------------------+
|                              Chat API (FastAPI)                               |
+---------------------------------------+----------------------------------------+
                                        |
                                        v
+--------------------------------------------------------------------------------+
|                               Chat Engine                                      |
|   Memory context · RAG-augmented system prompt · Tool loop · Persistence       |
+--+----------+----------+-----------+-----------+---------------+---------------+
   |          |          |           |           |               |
   v          v          v           v           v               v
+------+ +---------+ +--------+ +----------+ +-----------+ +------------+
|Memory| |RAG Eng. | |MCP Tool| |  Action  | |LLM Prov.  | | Response   |
|Mgr   | |multi-   | |Registry| | Executor | |OpenAI /   | | Evaluator  |
|      | |format   | |+ built-| |(approval)| |Anthropic /| | relevance, |
|      | |KB       | |ins     | |Jira etc. | |Ollama     | | groundedness|
+------+ +---------+ +--------+ +----------+ +-----------+ +------------+
   ^          ^          ^           ^           ^               ^
   |          |          |           |           |               |
   +----------+----------+-----------+-----------+---------------+
                                        |
                                        v
                               +----------------+
                               |   PostgreSQL   |
                               | Chroma / HTTP  |
                               | Redis + Celery |
                               +----------------+
```

**Request path (conceptual):** User → **Chat API (SSE)** → **Chat Engine** → **[Memory Manager, RAG Engine (PDF/DOCX/HTML/XLSX/CSV/TXT), MCP Tools, Action Executor (Jira / Email / Workflow)]** → **LLM** → **Response Evaluator** → User.

---

## RAG pipeline

1. **Ingest** — Uploads supported: **PDF** (PyMuPDF), **DOCX** (python-docx), **HTML** (BeautifulSoup), **XLSX** (openpyxl), **CSV**, **TXT**, and raw **text**. Text is normalized and optionally stripped of YAML front matter.
2. **Enterprise metadata** — Best-effort extraction of **department**, **version**, and **sensitivity** from HTML `<meta>`, DOCX core properties, PDF document info, CSV header row, and front matter (`department`, `version`, `sensitivity` / `classification`). Values are stored on `KnowledgeDocument.enterprise_metadata` and copied into chunk / vector metadata (Chroma-safe coercion).
3. **Chunk** — Per knowledge base: **fixed**, **sentence**, or **paragraph** strategy via LangChain `RecursiveCharacterTextSplitter` with configurable size and overlap.
4. **Embed** — `sentence-transformers` (default `all-MiniLM-L6-v2`); vectors in **ChromaDB** (embedded path or HTTP endpoint).
5. **Retrieve** — Cosine similarity; optional metadata filters; score threshold.
6. **Augment** — Numbered context blocks with citation hints `[1], [2], …`.
7. **Groundedness** — Lexical overlap heuristic plus structured **ResponseEvaluator** scores persisted per assistant turn.

---

## MCP integration

- Register MCP servers in Postgres (**stdio** command or **HTTP** URL).
- **Tool discovery** (`tools/list` / HTTP fallbacks); names exposed to the LLM as `mcp_{server_id}_{tool}`.
- Execution routes through **httpx** (HTTP) or subprocess JSON-RPC (stdio).
- **Built-in tools** (no MCP): `web_search`, `calculator`, `datetime`, `weather_stub`.

---

## Action execution & approval gates

**Action types:** `create_ticket`, `draft_email`, `compare_docs`, `summarize`, `create_task`, `initiate_workflow`.

- **Sensitive defaults** (`create_ticket`, `draft_email`, `initiate_workflow`) create a row in **`pending`** until an **admin** approves via `POST /api/actions/{id}/approve`. Override with `force_approval: true|false` in `params` when needed.
- **Non-gated** actions (`compare_docs`, `summarize`, `create_task`) start as **`approved`** and run on a **Celery** worker (`actions.run_execution`).
- Handlers: **Jira REST** (Basic auth: email + API token), **LLM-drafted email**, **LLM compare/summarize** over knowledge chunks, **in-memory task record**, **workflow webhook** (`WORKFLOW_WEBHOOK_URL` or per-request `webhook_url`).

Chat responses include **`suggested_actions`** (intent hints from the latest user message) so clients can surface one-click “run action” UX.

---

## Streaming (SSE)

- `POST /api/chat?chatbot_id=…` with `Accept: application/json` → full JSON body.
- Same endpoint with `stream: true` and `Accept: text/event-stream` → events: `start`, `token`, `done` (includes evaluation, sources, **suggested_actions**). Tool calling is disabled in stream mode to keep the protocol simple.

---

## Memory management

- **Short-term:** last N messages from PostgreSQL (`MAX_CONVERSATION_HISTORY`).
- **Long-term:** rolling summary and sliding-window strategies (see `MemoryManager` + Celery maintenance tasks).
- **Semantic recall:** embed query; retrieve similar past turns when available.

---

## Response evaluation pipeline

Scores every assistant message: **relevance**, **groundedness** (vs RAG sources), **helpfulness**, **safety**; persisted in `response_evaluations` and aggregated under `/api/analytics/quality`.

---

## Cloud deployment (reference)

| Concern | Option |
|--------|--------|
| **LLM** | Direct OpenAI/Anthropic/Ollama, or **AWS Bedrock** behind a compatible OpenAI-style proxy / adapter (swap `LLM_BASE_URL` + model IDs). |
| **Vectors** | Chroma HTTP today; **Amazon OpenSearch** k-NN / serverless vector collections with the same embedding dimension contract. |
| **Documents** | Local upload; production pattern: **S3** + presigned uploads, Celery workers pull object bytes into `RAGEngine.ingest_document`. |
| **Secrets** | AWS Secrets Manager / Parameter Store; never commit `.env`. |

---

## CI/CD

GitHub Actions workflow **`.github/workflows/ci.yml`** runs on push and pull request:

1. **Lint** — `ruff check app tests`
2. **Test** — `pytest`
3. **Docker** — `docker build` from the repo `Dockerfile` (image not pushed by default)

Extend with deploy jobs (ECS, Kubernetes, Fly.io, etc.) and environment-specific secrets.

---

## Observability

| Layer | Suggestion |
|-------|------------|
| **Metrics** | Expose Prometheus metrics from FastAPI (e.g. `prometheus-fastapi-instrumentator`) for request latency, RAG latency, Celery queue depth. |
| **Tracing** | OpenTelemetry SDK + OTLP exporter to Jaeger / AWS X-Ray / Grafana Tempo. |
| **Logs** | Structured JSON logging (request id, user id, chatbot id, conversation id); ship to CloudWatch / Loki / ELK. |

The codebase uses standard `logging` with configurable `LOG_LEVEL`.

---

## API endpoints (expanded)

| Area | Method & path |
|------|----------------|
| Auth | `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me` |
| Chat | `POST /api/chat`, `GET /api/chat/conversations`, `GET /api/chat/conversations/{id}`, `DELETE …`, `POST …/feedback` |
| Chatbots | CRUD `/api/chatbots`, `POST /api/chatbots/{id}/test` |
| Knowledge | CRUD `/api/kb`, `POST /api/kb/{id}/documents`, `GET /api/kb/{id}/search`, `POST /api/kb/{id}/reindex`, `GET /api/kb/{id}/stats` |
| **Actions** | **`POST /api/actions/execute`** (201 if auto-approved & queued, **202** if awaiting approval), **`GET /api/actions`**, **`GET /api/actions/{id}`**, **`POST /api/actions/{id}/approve`** (admin) |
| Prompts | CRUD `/api/prompts`, `POST /api/prompts/{id}/render` |
| MCP | CRUD `/api/mcp`, `POST /api/mcp/{id}/health`, `GET /api/mcp/{id}/tools`, `POST /api/mcp/tools/{tool_name}/test` |
| Analytics | `GET /api/analytics/usage`, `…/quality`, `…/models`, `…/popular-queries` |
| Health | `GET /health` |

---

## Tech stack (comprehensive)

- **Runtime:** Python 3.11+, FastAPI, Uvicorn  
- **Data:** SQLAlchemy 2 (async), PostgreSQL, Redis  
- **Tasks:** Celery  
- **LLM:** OpenAI SDK, Anthropic SDK, Ollama-compatible HTTP  
- **RAG:** LangChain text splitters, sentence-transformers, ChromaDB, PyMuPDF, python-docx, BeautifulSoup4, openpyxl  
- **Tools:** MCP client (stdio + HTTP), httpx  
- **Auth:** JWT, passlib/bcrypt  
- **Quality:** Custom heuristic evaluator + persisted scores  

---

## Project structure

```
app/
  main.py
  core/            # config, database, security, dependencies
  models/          # users, chatbots, RAG, MCP, evaluations, actions
  schemas/         # Pydantic I/O (incl. actions)
  api/routes/      # REST + SSE routers (incl. actions)
  services/        # llm_provider, rag_engine, memory_manager, chat_engine,
                   # evaluator, action_executor, mcp/
  tasks/           # Celery: indexing, maintenance, action execution
tests/
.github/workflows/
  ci.yml
```

---

## Scaling notes

- **Stateless API** behind a load balancer; SSE streams stay pinned to one instance or use a streaming-aware gateway.
- **Workers** for ingestion, re-embedding, and **action execution**; scale Celery concurrency independently.
- **Postgres:** connection pool tuning; read replicas for analytics.
- **Redis:** dedicated broker + result backend at scale.
- **Vector store:** Chroma HTTP → managed OpenSearch / Pinecone / Weaviate with the same embedding model contract.

---

## Run locally

```bash
cp .env.example .env
# Set SECRET_KEY, LLM_API_KEY, DATABASE_URL, VECTOR_DB_URL, optional JIRA_* / WORKFLOW_WEBHOOK_URL
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Celery worker (actions + maintenance):**

```bash
celery -A app.tasks.celery_app worker -l info
```

---

## Docker

```bash
export SECRET_KEY=$(openssl rand -hex 32)
export LLM_API_KEY=sk-...
docker build -t chatforge:latest .
docker run --rm -p 8000:8000 --env-file .env chatforge:latest
```

---

## Example: action execution

`POST /api/actions/execute` (Bearer token):

```json
{
  "action_type": "create_ticket",
  "params": {
    "title": "KB gap: refund policy",
    "description": "User could not find SLA for refunds.",
    "project": "SUPPORT",
    "priority": "High"
  }
}
```

If the action **requires approval**, response status is **202** and status is `pending` until an admin calls `POST /api/actions/{id}/approve` with `{ "approve": true }`. After approval, the worker runs the Jira call (or returns a **simulated** draft if `JIRA_*` env vars are unset).

---

## License

MIT (adjust for your portfolio).
