@echo off
REM Complete installation and setup batch file
REM Installs Ollama, pulls model, starts server, and runs app

echo ========================================================================
echo Complete Installation and Setup Script
echo Research Experiment Engine - Multi-Agent Multi-Platform Evaluator
echo ========================================================================
echo.

cd /d "%~dp0"

REM Step 1: Check Python
echo [Step 1/6] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10+ from: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo OK: Python is installed
echo.

REM Step 2: Install Python dependencies
echo [Step 2/6] Installing Python dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some dependencies may have failed to install
    echo Continuing anyway...
)
echo.

REM Step 3: Check Ollama
echo [Step 3/6] Checking Ollama installation...
ollama --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ollama is not installed. Attempting installation...
    echo.
    
    REM Try Winget
    where winget >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo Installing Ollama via Winget...
        winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
        if %ERRORLEVEL% EQU 0 (
            echo OK: Ollama installed via Winget
            echo Waiting for installation to complete...
            timeout /t 5 /nobreak >nul
            REM Refresh PATH
            call refreshenv >nul 2>&1
        ) else (
            echo Winget installation failed. Trying manual instructions...
            goto :install_manual
        )
    ) else (
        echo Winget not available. Manual installation required.
        goto :install_manual
    )
) else (
    ollama --version
    echo OK: Ollama is already installed
)
echo.

REM Step 4: Start Ollama server
echo [Step 4/6] Starting Ollama server...
REM Check if server is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo OK: Ollama server is already running
) else (
    echo Starting Ollama server in background...
    start /B ollama serve
    echo Waiting for server to start...
    timeout /t 5 /nobreak >nul
    
    REM Verify
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo OK: Ollama server started
    ) else (
        echo WARNING: Server may still be starting. Continuing...
    )
)
echo.

REM Step 5: Pull llama2 model
echo [Step 5/6] Checking llama2 model...
ollama list | findstr /C:"llama2" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo OK: llama2 model is already installed
) else (
    echo llama2 model not found. Installing...
    echo This will download ~3.8GB and may take several minutes...
    echo.
    ollama pull llama2
    if %ERRORLEVEL% EQU 0 (
        echo OK: llama2 model installed
    ) else (
        echo WARNING: Model installation failed or incomplete
        echo You can install manually later: ollama pull llama2
    )
)
echo.

REM Step 6: Run the app
echo [Step 6/6] Starting the application...
echo.
echo ========================================================================
echo Research Experiment Engine
echo ========================================================================
echo.
echo The app will now run all experiments.
echo This may take 20-40 minutes depending on your system.
echo.
echo ========================================================================
echo.

python app.py

echo.
echo ========================================================================
echo Application finished!
echo ========================================================================
echo.
echo Results are saved in: experiments\runs\<timestamp>\
echo.
pause
exit /b 0

:install_manual
echo.
echo ========================================================================
echo Manual Ollama Installation Required
echo ========================================================================
echo.
echo Please install Ollama manually:
echo   1. Download from: https://ollama.ai/download
echo   2. Run the installer (OllamaSetup.exe)
echo   3. Restart your terminal
echo   4. Run this script again
echo.
echo Opening download page...
start https://ollama.ai/download
echo.
echo Press any key to continue (app will run without Ollama)...
pause >nul
goto :eof

