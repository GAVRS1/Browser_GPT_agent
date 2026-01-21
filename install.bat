@echo off
echo Creating agent virtual environment...
python -m venv venv

echo.
echo Activating agent virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing agent requirements...
pip install -r requirements.txt

echo.
echo ==== Installing Playwright Chromium (agent) ====
playwright install chromium

echo.
echo Agent environment setup completed.
echo To start the bot, run "python main.py"
pause
