@echo off
cd /d F:\workspace\PVLDB\multi_agent_platform_evaluator

echo Initializing git repository...
call git init

echo Adding all files...
call git add .

echo Committing...
call git commit -m "first commit"

echo Setting main branch...
call git branch -M main

echo Adding remote...
call git remote add origin https://github.com/mahdinaser/multi-agent-platform-evaluator.git 2>nul
if errorlevel 1 (
    echo Remote may already exist, updating...
    call git remote set-url origin https://github.com/mahdinaser/multi-agent-platform-evaluator.git
)

echo Pushing to GitHub...
call git push -u origin main

if errorlevel 1 (
    echo.
    echo ========================================
    echo Push failed. This is usually due to:
    echo 1. Authentication required
    echo 2. Repository doesn't exist on GitHub yet
    echo.
    echo To fix:
    echo 1. Create repo at: https://github.com/new
    echo    Name: multi-agent-platform-evaluator
    echo 2. Authenticate: gh auth login
    echo 3. Run this script again
    echo ========================================
) else (
    echo.
    echo ========================================
    echo SUCCESS! Code pushed to GitHub!
    echo https://github.com/mahdinaser/multi-agent-platform-evaluator
    echo ========================================
)

pause

