@echo off
echo Creating MCP virtual environment...
python -m venv venv_mcp

echo.
echo Activating MCP virtual environment...
call venv_mcp\Scripts\activate.bat

echo.
echo Installing MCP requirements...
pip install -r requirements-mcp.txt

echo.
echo ==== Installing Playwright Python package (MCP) ====
pip install playwright

echo.
echo ==== Installing Playwright Chromium (MCP) ====
playwright install chromium

echo.
echo MCP environment setup completed.
