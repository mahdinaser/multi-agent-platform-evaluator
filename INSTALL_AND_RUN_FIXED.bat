@echo off
REM Fixed version - Complete installation and run script
REM Installs dependencies, Ollama, pulls model, starts server, runs app

title Research Experiment Engine - Complete Setup

echo.
echo ============================================================
echo  Research Experiment Engine
echo  Complete Installation and Setup
echo ============================================================
echo.

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo [1/5] Installing Python packages...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some packages may have failed to install
)
echo Done.
echo.

echo [2/5] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo Ollama not found. Attempting installation...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo Winget not available. Manual installation required.
        echo Please install from: https://ollama.ai/download
        echo App will run without Ollama.
        echo.
        timeout /t 3 >nul
    ) else (
        echo Installing Ollama via Winget...
        winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements --silent
        if errorlevel 1 (
            echo [WARNING] Could not auto-install Ollama.
            echo Please install from: https://ollama.ai/download
            echo App will run without Ollama (LLM agent will use fallback).
            echo.
            timeout /t 3 >nul
        ) else (
            echo Ollama installed. Waiting for setup...
            timeout /t 5 >nul
        )
    )
) else (
    ollama --version
    echo Ollama found.
)
echo.

echo [3/5] Starting Ollama server...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Starting server...
    start /B ollama serve >nul 2>&1
    timeout /t 3 >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo Server may still be starting...
    ) else (
        echo Server running.
    )
) else (
    echo Server already running.
)
echo.

echo [4/5] Checking llama2 model...
ollama list 2>nul | findstr /C:"llama2" >nul
if errorlevel 1 (
    echo Model not found. Installing llama2...
    echo This will download ~3.8GB and may take several minutes...
    ollama pull llama2
    if errorlevel 1 (
        echo [WARNING] Model installation failed or incomplete.
        echo You can install manually: ollama pull llama2
    ) else (
        echo Model installed.
    )
) else (
    echo Model found.
)
echo.

echo [5/5] Running application...
echo.
echo ============================================================
echo  Starting Experiments
echo ============================================================
echo.

python app.py

echo.
echo ============================================================
echo  Complete!
echo ============================================================
echo.
pause

