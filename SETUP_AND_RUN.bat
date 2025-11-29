@echo off
REM Complete setup and run script - Fixed version
REM Installs everything and runs the app

setlocal enabledelayedexpansion

title Research Experiment Engine - Complete Setup

echo.
echo ============================================================
echo  Research Experiment Engine
echo  Complete Installation and Setup
echo ============================================================
echo.

cd /d "%~dp0"

REM Step 1: Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10+ first.
    pause
    exit /b 1
)
python --version
echo.

REM Step 2: Install Python packages
echo [2/5] Installing Python packages...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some packages may have failed to install
)
echo Done.
echo.

REM Step 3: Check/Install Ollama
echo [3/5] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo Ollama not found. Attempting installation...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo Winget not available.
        echo Please install Ollama from: https://ollama.ai/download
        echo App will run without Ollama.
        echo.
        timeout /t 3 >nul
        set OLLAMA_AVAILABLE=0
    ) else (
        echo Installing via Winget...
        winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements --silent
        if errorlevel 1 (
            echo [WARNING] Auto-install failed.
            echo Please install manually from: https://ollama.ai/download
            echo App will run without Ollama.
            set OLLAMA_AVAILABLE=0
        ) else (
            echo Ollama installed. Waiting...
            timeout /t 5 >nul
            set OLLAMA_AVAILABLE=1
        )
    )
) else (
    ollama --version
    echo Ollama found.
    set OLLAMA_AVAILABLE=1
)
echo.

REM Step 4: Start server and pull model (if Ollama available)
if "!OLLAMA_AVAILABLE!"=="1" (
    echo [4/5] Setting up Ollama server and model...
    
    REM Check if server running
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo Starting server...
        start /B ollama serve
        timeout /t 3 >nul
    ) else (
        echo Server already running.
    )
    
    REM Check model
    ollama list 2>nul | findstr /C:"llama2" >nul
    if errorlevel 1 (
        echo Installing llama2 model (~3.8GB, may take time)...
        ollama pull llama2
    ) else (
        echo Model found.
    )
) else (
    echo [4/5] Skipping Ollama setup (not available)
)
echo.

REM Step 5: Run app
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

endlocal

