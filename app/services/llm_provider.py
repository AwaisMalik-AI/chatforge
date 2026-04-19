"""Multi-provider LLM abstraction with streaming and tool calling."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


def _openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return messages


def _anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system: str | None = None
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system = (system or "") + (m.get("content") or "")
            continue
        if role in ("user", "assistant"):
            out.append({"role": role, "content": m.get("content") or ""})
    return system, out


class LLMProvider:
    def __init__(self) -> None:
        self._openai: AsyncOpenAI | None = None
        self._anthropic: AsyncAnthropic | None = None

    def _openai_client(self) -> AsyncOpenAI:
        if self._openai is None:
            self._openai = AsyncOpenAI(
                api_key=settings.LLM_API_KEY or "",
                base_url=settings.LLM_BASE_URL or None,
            )
        return self._openai

    def _anthropic_client(self) -> AsyncAnthropic:
        if self._anthropic is None:
            if not settings.LLM_API_KEY:
                raise RuntimeError("LLM_API_KEY is required for Anthropic")
            self._anthropic = AsyncAnthropic(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL or None,
            )
        return self._anthropic

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncGenerator[str, None]:
        prov = (provider or settings.LLM_PROVIDER).lower()
        mdl = model or settings.LLM_MODEL
        temp = temperature if temperature is not None else settings.TEMPERATURE
        mtok = max_tokens if max_tokens is not None else settings.MAX_TOKENS
        if stream:
            return self.stream_chat(
                messages,
                model=mdl,
                provider=prov,
                temperature=temp,
                max_tokens=mtok,
                tools=tools,
            )
        if prov == "openai":
            return await self._openai_chat(messages, mdl, temp, mtok, tools)
        if prov == "anthropic":
            return await self._anthropic_chat(messages, mdl, temp, mtok, tools)
        if prov == "ollama":
            return await self._ollama_chat(messages, mdl, temp, mtok, tools)
        raise ValueError(f"Unsupported LLM_PROVIDER: {prov}")

    async def _openai_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        client = self._openai_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": _openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = await client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return {
            "content": content,
            "tool_calls": tool_calls,
            "model": getattr(resp, "model", model),
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        }

    async def _anthropic_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        client = self._anthropic_client()
        system, msgs = _anthropic_messages(messages)
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for t in tools:
                if t.get("type") == "function":
                    fn = t.get("function", {})
                    anthropic_tools.append(
                        {
                            "name": fn.get("name"),
                            "description": fn.get("description", ""),
                            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                        }
                    )
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        resp = await client.messages.create(**kwargs)
        content = ""
        tool_calls = None
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content += getattr(block, "text", "") or ""
            elif btype == "tool_use":
                tool_calls = tool_calls or []
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                            if isinstance(block.input, dict)
                            else str(block.input),
                        },
                    }
                )
        return {
            "content": content,
            "tool_calls": tool_calls,
            "model": model,
            "usage": {
                "prompt_tokens": resp.usage.input_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.output_tokens if resp.usage else 0,
            },
        }

    async def _ollama_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        base = (settings.LLM_BASE_URL or "http://localhost:11434").rstrip("/")
        url = f"{base}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls = choice.get("tool_calls")
        usage = data.get("usage") or {}
        return {
            "content": content,
            "tool_calls": tool_calls,
            "model": data.get("model", model),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        }

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[str, None]:
        prov = (provider or settings.LLM_PROVIDER).lower()
        mdl = model or settings.LLM_MODEL
        temp = temperature if temperature is not None else settings.TEMPERATURE
        mtok = max_tokens if max_tokens is not None else settings.MAX_TOKENS
        if prov == "openai" or prov == "ollama":
            async for chunk in self._stream_openai_compatible(messages, mdl, temp, mtok, tools, prov):
                yield chunk
            return
        if prov == "anthropic":
            async for chunk in self._stream_anthropic(messages, mdl, temp, mtok, tools):
                yield chunk
            return
        raise ValueError(f"Streaming not supported for provider: {prov}")

    async def _stream_openai_compatible(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        prov: str,
    ) -> AsyncGenerator[str, None]:
        if prov == "openai":
            client = self._openai_client()
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": _openai_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if tools:
                kwargs["tools"] = tools
            stream = await client.chat.completions.create(**kwargs)
            async for event in stream:
                delta = event.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        else:
            base = (settings.LLM_BASE_URL or "http://localhost:11434").rstrip("/")
            url = f"{base}/v1/chat/completions"
            body = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if tools:
                body["tools"] = tools
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload)
                            delta = obj["choices"][0].get("delta", {})
                            c = delta.get("content")
                            if c:
                                yield c
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    async def _stream_anthropic(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
    ) -> AsyncGenerator[str, None]:
        client = self._anthropic_client()
        system, msgs = _anthropic_messages(messages)
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for t in tools:
                if t.get("type") == "function":
                    fn = t.get("function", {})
                    anthropic_tools.append(
                        {
                            "name": fn.get("name"),
                            "description": fn.get("description", ""),
                            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                        }
                    )
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msgs,
            "stream": True,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str, model: str | None = None) -> int:
        mdl = model or settings.LLM_MODEL
        try:
            enc = tiktoken.encoding_for_model(mdl)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))


llm_provider = LLMProvider()
