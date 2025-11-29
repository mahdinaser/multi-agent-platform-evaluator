@echo off
echo ========================================
echo Pushing to GitHub
echo ========================================
echo.

cd /d F:\workspace\PVLDB\multi_agent_platform_evaluator

echo Step 1: Initializing git...
git init
if errorlevel 1 (
    echo ERROR: Git init failed
    pause
    exit /b 1
)

echo.
echo Step 2: Adding files...
git add README.md .gitignore
git add agents/ platforms/ src/ config/ app.py requirements.txt
git add *.md
git add report/*.tex report/*.py report/*.md report/Makefile report/compile.bat 2>nul

echo.
echo Step 3: Checking status...
git status

echo.
echo Step 4: Committing...
git commit -m "Initial commit: Multi-agent platform evaluator framework"
if errorlevel 1 (
    echo WARNING: Commit failed - may already be committed or no changes
)

echo.
echo Step 5: Setting main branch...
git branch -M main

echo.
echo Step 6: Adding remote...
git remote remove origin 2>nul
git remote add origin https://github.com/mahdinaser/multi-agent-platform-evaluator.git

echo.
echo Step 7: Pushing to GitHub...
echo NOTE: You may need to authenticate (GitHub CLI, SSH, or Personal Access Token)
git push -u origin main

if errorlevel 1 (
    echo.
    echo ERROR: Push failed. Common issues:
    echo   1. Authentication required - use GitHub CLI, SSH, or Personal Access Token
    echo   2. Repository may not exist on GitHub - create it first at:
    echo      https://github.com/mahdinaser/multi-agent-platform-evaluator
    echo   3. Check your git credentials
    echo.
    echo To authenticate with GitHub CLI:
    echo   gh auth login
    echo   git push -u origin main
    echo.
) else (
    echo.
    echo SUCCESS! Code pushed to GitHub!
    echo Repository: https://github.com/mahdinaser/multi-agent-platform-evaluator
    echo.
)

pause

