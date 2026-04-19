# ChatForge — Internal Guide

> This file is for personal reference and learning. Do NOT upload to GitHub.

## What This Project Is

A production LLM chatbot platform with multi-tenant chatbots, RAG knowledge bases, SSE streaming responses, conversation memory management (short-term sliding window + long-term summarization), MCP (Model Context Protocol) tool integration, prompt templates, response quality evaluation, and analytics.

## Why I Built This

Demonstrates the most in-demand AI engineering skills:
- LLM integration with multiple providers (OpenAI, Anthropic, Ollama)
- RAG pipeline: chunking strategies, embeddings, semantic search, grounded responses
- MCP tool integration (Model Context Protocol — the emerging standard for AI tool use)
- Streaming architecture with Server-Sent Events (SSE)
- Conversation memory: sliding window for recent context, LLM summarization for long-term
- Response evaluation: relevance, groundedness, helpfulness, safety scoring
- Multi-tenant chatbot management with per-bot configuration

## How to Explain to Recruiters

**Elevator pitch**: "I built a production chatbot platform where you can create AI chatbots backed by custom knowledge bases using RAG. It supports streaming responses via SSE, connects to external tools through MCP (Model Context Protocol), manages conversation memory with a sliding window plus summarization strategy, and evaluates every response for quality and groundedness. It works with OpenAI, Anthropic, or local Ollama models."

**Technical depth points**:

1. **RAG Pipeline**: Documents chunked with configurable strategies (fixed/sentence/paragraph), embedded, stored in ChromaDB. Queries get semantic search with score thresholds. Responses include source citations to prevent hallucination. Groundedness evaluation scores how well the response matches retrieved sources.

2. **MCP Integration**: Implements MCP client for both stdio (subprocess) and HTTP transports. MCP servers register their tools, which get formatted as LLM function calls. When the LLM requests a tool call, the system routes it to the correct MCP server, executes it, feeds the result back, and continues the conversation.

3. **Streaming (SSE)**: Uses Server-Sent Events for real-time token streaming. Each chunk is pushed as an SSE event. The client gets tokens as they're generated, not waiting for the full response. Works with all three LLM providers.

4. **Memory Management**: Two-tier system. Short-term: sliding window of last N messages sent with each request. Long-term: older messages summarized by the LLM into a condensed context block. Semantic memory: retrieve relevant past messages based on query similarity.

5. **Response Evaluation**: Every response scored on four dimensions — relevance (to the query), groundedness (in retrieved sources), helpfulness (completeness), and safety. Scores tracked over time for quality monitoring.

## Key Design Decisions

- **Provider abstraction**: Same chatbot can work with OpenAI, Anthropic, or Ollama by changing one config field. Each provider's tool calling format handled internally.
- **Chatbot isolation**: Each chatbot has its own system prompt, knowledge bases, tools, and model configuration. Multi-tenant by design.
- **MCP over custom tool implementations**: MCP is the industry standard (backed by Anthropic). Building MCP support shows awareness of the ecosystem direction.
- **ChromaDB for dev, swappable for production**: Vector store is abstracted enough to swap to Pinecone/Weaviate.

## Common Interview Questions

**Q: What is MCP and why did you use it?**
A: Model Context Protocol is a standard for connecting AI models to external tools and data sources. Instead of hardcoding tool implementations, MCP lets you connect to any MCP-compatible server that exposes tools. My platform discovers tools automatically when you register an MCP server, formats them for the LLM's function calling, and routes execution. This makes the system extensible without code changes.

**Q: How does your RAG prevent hallucination?**
A: Three mechanisms — (1) retrieved sources are injected into the prompt with explicit instructions to only use provided context, (2) the system prompt tells the LLM to say "I don't know" when sources don't cover the question, (3) a post-generation groundedness evaluation scores how well the response matches the sources. Low groundedness scores trigger alerts.

**Q: SSE vs WebSockets for streaming — why SSE?**
A: SSE is simpler, HTTP-based, works through proxies and CDNs natively, auto-reconnects, and is sufficient for our use case (server-to-client streaming). WebSockets would only be needed if we needed bidirectional real-time communication. OpenAI's own API uses SSE.

**Q: How would you scale this to 10K concurrent conversations?**
A: API is stateless — scale horizontally. Conversation memory in PostgreSQL with connection pooling. ChromaDB swap to managed Pinecone with serverless scaling. Redis for session caching. LLM calls are the bottleneck — implement request queuing with priority, response caching for repeated queries, and Ollama clusters for self-hosted models.
