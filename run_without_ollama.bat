@echo off
echo ============================================================
echo Running App Without Ollama (Using Fallback)
echo ============================================================
echo.
echo NOTE: Ollama is not installed.
echo The LLM agent will use fallback reasoning (rule-based).
echo.
echo To use real LLM:
echo   1. Install Ollama from: https://ollama.ai/download
echo   2. Run: run_ollama_and_app.bat
echo.
echo ============================================================
echo.

cd /d "%~dp0"
python app.py

pause

