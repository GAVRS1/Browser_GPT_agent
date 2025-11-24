@echo off
echo ==== Checking virtual environment ====

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo ==== Activating virtual environment ====
call venv\Scripts\activate.bat

echo ==== Starting Browser AI Agent ====
python main.py

echo ==== Finished ====
pause
