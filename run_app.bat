@echo off
cd /d "%~dp0"
echo Running app.py...
echo.
python app.py
echo.
echo Exit code: %ERRORLEVEL%
pause

