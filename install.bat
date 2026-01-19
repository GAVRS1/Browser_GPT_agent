@echo off
echo ==== Installing agent environment ====
call install_agent.bat

echo.
echo ==== Installing MCP environment ====
call install_mcp.bat

echo.
echo Installation completed!
echo To start the bot, run "python main.py"
pause
