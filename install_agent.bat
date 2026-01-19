@echo off
echo Creating agent virtual environment...
python -m venv venv

echo.
echo Activating agent virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing agent requirements...
pip install -r requirements-agent.txt

echo.
echo ==== Installing Playwright Python package (agent) ====
pip install playwright

echo.
echo ==== Installing Playwright Chromium (agent) ====
playwright install chromium

echo.
echo Agent environment setup completed.
