from __future__ import annotations

import json
from typing import Any, Dict, Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP

from agent.browser_tools import BrowserToolbox, ToolSchema, ToolResult


def _result_to_payload(result: ToolResult) -> str:
    payload = {
        "name": result.name,
        "success": result.success,
        "observation": result.observation,
    }
    return json.dumps(payload, ensure_ascii=False)


def _register_tool(mcp: FastMCP, toolbox: BrowserToolbox, schema: ToolSchema) -> None:
    tool_name = schema.name

    @mcp.tool(name=tool_name, description=schema.description, input_schema=schema.parameters)
    def _tool_handler(**kwargs: Any) -> str:
        result = toolbox.execute(tool_name, kwargs)
        if not result.success:
            logger.warning(f"[mcp] Tool {tool_name} failed: {result.observation}")
        return _result_to_payload(result)


def create_mcp_server(toolbox: Optional[BrowserToolbox] = None) -> FastMCP:
    mcp = FastMCP("browser-gpt-agent")
    toolbox = toolbox or BrowserToolbox()

    for schema in toolbox.tool_schemas():
        _register_tool(mcp, toolbox, schema)

    logger.info("[mcp] Browser tools registered.")
    return mcp


def run() -> None:
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    run()
