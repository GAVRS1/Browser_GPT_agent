from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPToolCallResult:
    success: bool
    observation: str


def _extract_text_content(content: Iterable[Any]) -> str:
    parts: List[str] = []
    for item in content or []:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content")
        else:
            text = getattr(item, "text", None) or getattr(item, "content", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _parse_payload(text: str) -> Tuple[bool, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return True, text
    if not isinstance(payload, dict):
        return True, text
    observation = payload.get("observation", text)
    success = bool(payload.get("success", True))
    return success, str(observation)


class MCPToolClient:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._session: Optional[ClientSession] = None
        self._stdio_ctx = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def start(self) -> None:
        if self._session is not None:
            return
        self._run(self._connect())

    async def _connect(self) -> None:
        command = os.getenv("MCP_SERVER_COMMAND", sys.executable).strip() or sys.executable
        raw_args = os.getenv("MCP_SERVER_ARGS", "-m agent.mcp_server")
        args = shlex.split(raw_args)
        env = os.environ.copy()

        server_params = StdioServerParameters(command=command, args=args, env=env)
        self._stdio_ctx = stdio_client(server_params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        logger.info("[mcp] Client session initialized.")

    def close(self) -> None:
        if self._session is None:
            return
        self._run(self._disconnect())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)

    async def _disconnect(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._stdio_ctx is not None:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None
        self._tools_cache = None
        logger.info("[mcp] Client session closed.")

    def list_tools(self) -> List[Dict[str, Any]]:
        self.start()
        if self._tools_cache is not None:
            return list(self._tools_cache)
        tools = self._run(self._session.list_tools())
        normalized = []
        for tool in tools:
            if isinstance(tool, dict):
                normalized.append(tool)
            else:
                normalized.append(
                    {
                        "name": getattr(tool, "name", ""),
                        "description": getattr(tool, "description", ""),
                        "input_schema": getattr(tool, "input_schema", {}),
                    }
                )
        self._tools_cache = normalized
        return list(normalized)

    def openai_tools(self) -> List[Dict[str, Any]]:
        tools = self.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in tools
        ]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> MCPToolCallResult:
        self.start()
        args = arguments or {}
        result = self._run(self._session.call_tool(name, args))
        raw_content = _extract_text_content(getattr(result, "content", None) or [])
        success, observation = _parse_payload(raw_content)
        is_error = bool(getattr(result, "is_error", False))
        return MCPToolCallResult(success and not is_error, observation)
