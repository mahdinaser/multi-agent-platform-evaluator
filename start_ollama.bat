@echo off
echo Starting Ollama server...
echo.

REM Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not installed or not in PATH
    echo.
    echo Please install Ollama from: https://ollama.ai/
    echo After installation, add Ollama to your PATH or restart your terminal
    pause
    exit /b 1
)

REM Check if Ollama is already running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Ollama server is already running!
    ollama list
    pause
    exit /b 0
)

REM Start Ollama server in background
echo Starting Ollama server...
start /B ollama serve

REM Wait for server to start
timeout /t 3 /nobreak >nul

REM Check if server started
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Ollama server started successfully!
    echo.
    echo Available models:
    ollama list
    echo.
    echo Server is running on http://localhost:11434
    echo.
    echo To stop the server, close this window or run: taskkill /F /IM ollama.exe
) else (
    echo.
    echo WARNING: Could not verify Ollama server is running
    echo Please check if Ollama is installed correctly
    echo.
)

pause

