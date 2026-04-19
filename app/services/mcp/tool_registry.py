"""Tool registry: MCP servers + built-in tools for LLM function calling."""

from __future__ import annotations

import ast
import logging
import operator
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chatbot import Chatbot, MCPServerConfig
from app.services.mcp.mcp_client import mcp_client

logger = logging.getLogger(__name__)

_BUILTIN_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo HTML (lite). Returns snippet text.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate a numeric arithmetic expression (+ - * / ** parentheses).",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "datetime",
            "description": "Return current UTC ISO timestamp.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather_stub",
            "description": "Stub weather lookup (returns placeholder; wire to a real API in production).",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
]


class ToolRegistry:
    def __init__(self) -> None:
        self._mcp_tools_cache: dict[int, list[dict[str, Any]]] = {}

    def _builtin_names(self) -> set[str]:
        return {t["function"]["name"] for t in _BUILTIN_SPECS}

    async def register_server(self, db: AsyncSession, config: MCPServerConfig) -> list[dict[str, Any]]:
        tools = await mcp_client.list_tools(config.id, config)
        config.tools_available = tools
        self._mcp_tools_cache[config.id] = tools
        await db.flush()
        return tools

    def _openai_tool_from_mcp(self, t: dict[str, Any], server_id: int) -> dict[str, Any]:
        name = t.get("name", "unknown")
        desc = t.get("description", "")
        schema = t.get("inputSchema") or t.get("input_schema") or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": f"mcp_{server_id}_{name}",
                "description": f"[MCP server {server_id}] {desc}",
                "parameters": schema,
            },
        }

    async def get_tools_for_chatbot(self, db: AsyncSession, chatbot_id: int) -> list[dict[str, Any]]:
        bot = await db.get(Chatbot, chatbot_id)
        if not bot:
            return []
        enabled = list(bot.tools_enabled or [])
        if not enabled:
            return []
        out: list[dict[str, Any]] = []
        builtins = self._builtin_names()
        seen: set[str] = set()
        for item in enabled:
            if isinstance(item, str) and item in builtins:
                spec = next(s for s in _BUILTIN_SPECS if s["function"]["name"] == item)
                if item not in seen:
                    out.append(spec)
                    seen.add(item)
                continue
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit()):
                sid = int(item)
                cfg = await db.get(MCPServerConfig, sid)
                if not cfg or not cfg.is_active:
                    continue
                tools = cfg.tools_available or self._mcp_tools_cache.get(sid) or []
                if not tools:
                    tools = await mcp_client.list_tools(sid, cfg)
                    cfg.tools_available = tools
                for t in tools:
                    prefixed = self._openai_tool_from_mcp(t, sid)
                    fn = prefixed["function"]["name"]
                    if fn not in seen:
                        out.append(prefixed)
                        seen.add(fn)
                continue
            if str(item) == "all_builtins":
                for spec in _BUILTIN_SPECS:
                    n = spec["function"]["name"]
                    if n not in seen:
                        out.append(spec)
                        seen.add(n)
        return out

    async def _web_search(self, query: str) -> str:
        url = "https://lite.duckduckgo.com/lite/"
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            r = await client.post(url, data={"q": query})
            r.raise_for_status()
            text = r.text[:4000]
        return text

    def _safe_calc(self, expression: str) -> float:
        node = ast.parse(expression, mode="eval")

        def _eval(n: ast.AST) -> float:
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, int | float):
                return float(n.value)
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
                return -_eval(n.operand)
            if isinstance(n, ast.BinOp):
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                }
                op = ops.get(type(n.op))
                if not op:
                    raise ValueError("unsupported op")
                return op(_eval(n.left), _eval(n.right))
            raise ValueError("unsupported syntax")

        return _eval(node)

    async def execute_tool(self, db: AsyncSession, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "web_search":
            q = arguments.get("query", "")
            return await self._web_search(q)
        if tool_name == "calculator":
            expr = arguments.get("expression", "0")
            try:
                return str(self._safe_calc(expr))
            except Exception as e:
                return f"Error: {e}"
        if tool_name == "datetime":
            return datetime.now(UTC).isoformat()
        if tool_name == "weather_stub":
            loc = arguments.get("location", "unknown")
            return {
                "location": loc,
                "note": "stub",
                "conditions": "n/a — configure a weather API in production",
            }
        if tool_name.startswith("mcp_"):
            parts = tool_name.split("_", 2)
            if len(parts) < 3:
                return {"error": "invalid mcp tool name"}
            server_id = int(parts[1])
            real_name = parts[2]
            cfg = await db.get(MCPServerConfig, server_id)
            if not cfg:
                return {"error": "server not found"}
            return await mcp_client.call_tool(server_id, cfg, real_name, arguments)
        return {"error": f"unknown tool: {tool_name}"}


tool_registry = ToolRegistry()
