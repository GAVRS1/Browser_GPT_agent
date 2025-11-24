@echo off
echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing requirements...
pip install -r requirements.txt
echo.
echo ==== Installing Playwright Python package ====
pip install playwright

echo.
echo ==== Installing Playwright browsers ====
playwright install
echo.
echo Installation completed!
echo To start the bot, run "python main.py"
pause
