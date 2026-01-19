from __future__ import annotations

import json
from typing import Any, Optional

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

    # FastMCP.tool() не поддерживает input_schema / inputSchema.
    # Поэтому просто регистрируем tool, а JSON schema (если нужно) добавляем в описание.
    description = schema.description or ""
    if schema.parameters:
        try:
            description = (
                description
                + "\n\n[Input JSON Schema]\n"
                + json.dumps(schema.parameters, ensure_ascii=False)
            )
        except Exception:
            # если вдруг schema.parameters не сериализуется — не ломаем регистрацию
            pass

    @mcp.tool(name=tool_name, description=description)
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

    import asyncio
    import threading
    from loguru import logger

    # Если в текущем потоке уже крутится event loop (Playwright sync API часто так делает),
    # FastMCP.run() упадёт. Запускаем MCP в отдельном потоке.
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if loop_running:
        logger.warning("[mcp] Detected running asyncio loop; starting FastMCP stdio server in a dedicated thread.")
        t = threading.Thread(target=server.run, name="mcp-stdio", daemon=False)
        t.start()
        t.join()
    else:
        server.run()


if __name__ == "__main__":
    run()
