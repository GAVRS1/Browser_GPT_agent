from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.browser_tools import BrowserToolbox


@dataclass
class ToolCallResult:
    success: bool
    observation: str


class ToolClient:
    def __init__(self) -> None:
        self._toolbox = BrowserToolbox()

    def list_tools(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for schema in self._toolbox.tool_schemas():
            tools.append(
                {
                    "name": schema.name,
                    "description": schema.description,
                    "input_schema": schema.parameters,
                }
            )
        return tools

    def openai_tools(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for schema in self._toolbox.tool_schemas():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.parameters,
                    },
                }
            )
        return tools

    def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> ToolCallResult:
        result = self._toolbox.execute(name, arguments)
        return ToolCallResult(result.success, result.observation)


_shared_client: Optional[ToolClient] = None


def get_shared_tool_client() -> ToolClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = ToolClient()
    return _shared_client


def close_shared_tool_client() -> None:
    global _shared_client
    _shared_client = None
