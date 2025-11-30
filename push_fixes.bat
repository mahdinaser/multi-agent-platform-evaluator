@echo off
cd /d F:\workspace\PVLDB\multi_agent_platform_evaluator

echo ========================================
echo Pushing fixes to GitHub
echo ========================================
echo.

echo Step 1: Adding all changes...
git add -A

echo.
echo Step 2: Checking status...
git status --short

echo.
echo Step 3: Committing changes...
git commit -m "Fix: Add get_decision_reasoning() method to all agents"

if errorlevel 1 (
    echo WARNING: Commit failed - may be no changes or already committed
    git status
)

echo.
echo Step 4: Pushing to GitHub...
git push origin main

if errorlevel 1 (
    echo.
    echo ERROR: Push failed. Common issues:
    echo   1. Authentication required
    echo   2. Network issues
    echo   3. Repository permissions
    echo.
    echo Try: git push origin main
) else (
    echo.
    echo ========================================
    echo SUCCESS! Changes pushed to GitHub!
    echo ========================================
    echo.
    echo Repository: https://github.com/mahdinaser/multi-agent-platform-evaluator
    echo.
)

pause

