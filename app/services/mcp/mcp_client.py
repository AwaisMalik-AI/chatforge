"""MCP client: stdio (JSON-RPC subprocess) and HTTP transport."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import uuid
from typing import Any

import httpx

from app.models.chatbot import MCPServerConfig, MCPServerType

logger = logging.getLogger(__name__)


class MCPClient:
    """Minimal MCP-style client for tool discovery and invocation."""

    def __init__(self) -> None:
        self._stdio_locks: dict[int, asyncio.Lock] = {}

    async def connect(self, server_config: MCPServerConfig) -> bool:
        if server_config.server_type == MCPServerType.http:
            return await self.health_check(server_config.id, server_config)
        if server_config.server_type == MCPServerType.stdio:
            return bool(server_config.command)
        return False

    def _lock(self, server_id: int) -> asyncio.Lock:
        if server_id not in self._stdio_locks:
            self._stdio_locks[server_id] = asyncio.Lock()
        return self._stdio_locks[server_id]

    async def _stdio_rpc(
        self,
        server_config: MCPServerConfig,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        if not server_config.command:
            raise ValueError("stdio server requires command")
        cmd_parts = shlex.split(server_config.command, posix=False)
        if not cmd_parts:
            raise ValueError("empty command")
        req = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }
        payload = json.dumps(req) + "\n"
        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin
        proc.stdin.write(payload.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            raise
        if proc.returncode not in (0, None):
            logger.warning("MCP stdio rc=%s stderr=%s", proc.returncode, err.decode()[:500])
        line = out.decode().strip().splitlines()[0] if out else "{}"
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"error": {"message": line[:500]}, "result": None}

    async def list_tools(
        self,
        server_id: int,
        server_config: MCPServerConfig,
    ) -> list[dict[str, Any]]:
        if server_config.server_type == MCPServerType.http:
            base = (server_config.url or "").rstrip("/")
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{base}/tools/list",
                    json={},
                )
                if r.status_code >= 400:
                    r = await client.get(f"{base}/mcp/tools")
                r.raise_for_status()
                data = r.json()
            tools = data.get("tools") or data.get("result", {}).get("tools") or []
            return tools if isinstance(tools, list) else []
        async with self._lock(server_id):
            resp = await self._stdio_rpc(server_config, "tools/list", {})
        if resp.get("error"):
            logger.error("MCP tools/list error: %s", resp["error"])
            return []
        result = resp.get("result") or {}
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    async def call_tool(
        self,
        server_id: int,
        server_config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if server_config.server_type == MCPServerType.http:
            base = (server_config.url or "").rstrip("/")
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"{base}/tools/call",
                    json={"name": tool_name, "arguments": arguments},
                )
                r.raise_for_status()
                return r.json()
        async with self._lock(server_id):
            resp = await self._stdio_rpc(
                server_config,
                "tools/call",
                {"name": tool_name, "arguments": arguments},
            )
        if resp.get("error"):
            return {"isError": True, "content": [{"type": "text", "text": str(resp["error"])}]}
        return resp.get("result") or {"content": []}

    async def health_check(self, server_id: int, server_config: MCPServerConfig | None = None) -> bool:
        if server_config is None:
            return False
        try:
            if server_config.server_type == MCPServerType.http:
                base = (server_config.url or "").rstrip("/")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.get(f"{base}/health")
                    return r.status_code < 400
            if server_config.server_type == MCPServerType.stdio:
                return bool(server_config.command)
        except Exception as e:
            logger.debug("health_check failed: %s", e)
        return False


mcp_client = MCPClient()
