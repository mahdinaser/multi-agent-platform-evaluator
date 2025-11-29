@echo off
echo ============================================================
echo Starting Ollama Server and App in Separate Windows
echo ============================================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not installed!
    echo.
    echo Please install Ollama first:
    echo   1. Download from: https://ollama.ai/download
    echo   2. Install and restart terminal
    echo   3. Run this script again
    echo.
    pause
    exit /b 1
)

echo [1/3] Starting Ollama server in new window...
start "Ollama Server" cmd /k "echo Ollama Server Terminal && echo ======================================== && echo Keep this window open! && echo ======================================== && ollama serve"

echo [2/3] Waiting for server to start...
timeout /t 5 /nobreak >nul

echo [3/3] Starting app.py in new window...
cd /d "%~dp0"
start "Research Experiment Engine" cmd /k "cd /d %CD% && echo Research Experiment Engine && echo ======================================== && python app.py && echo. && echo Press any key to exit... && pause >nul"

echo.
echo ============================================================
echo Setup Complete!
echo.
echo Two windows are now open:
echo   1. Ollama Server - Keep this open!
echo   2. App Terminal - Running experiments
echo.
echo The app will use Ollama for LLM agent decisions.
echo ============================================================
echo.
pause

