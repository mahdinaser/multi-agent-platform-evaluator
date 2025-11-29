@echo off
REM Quick install and run - simplest version

title Quick Install and Run

cd /d "%~dp0"

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting app...
echo (Ollama will be used if available, otherwise fallback)
echo.

python app.py

pause

