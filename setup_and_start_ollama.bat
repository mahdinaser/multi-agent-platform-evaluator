@echo off
REM Batch file wrapper for PowerShell script
echo Running Ollama Setup Script...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0setup_and_start_ollama.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Setup encountered errors. Check the output above.
    pause
    exit /b %ERRORLEVEL%
)

