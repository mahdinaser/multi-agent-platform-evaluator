# Push to GitHub - Commands to Run

## Step 1: Navigate to project directory
```bash
cd F:\workspace\PVLDB\multi_agent_platform_evaluator
```

## Step 2: Initialize Git (if not already done)
```bash
git init
```

## Step 3: Add all files
```bash
# Add README and gitignore first
git add README.md .gitignore

# Add all source code
git add agents/ platforms/ src/ config/ app.py requirements.txt

# Add documentation
git add *.md

# Add report files (optional - you may want to exclude PDFs)
git add report/*.tex report/*.py report/*.md report/Makefile report/compile.bat

# Or add everything (except what's in .gitignore)
git add .
```

## Step 4: Commit
```bash
git commit -m "Initial commit: Multi-agent platform evaluator framework"
```

## Step 5: Set main branch
```bash
git branch -M main
```

## Step 6: Add remote
```bash
git remote add origin https://github.com/mahdinaser/multi-agent-platform-evaluator.git
```

## Step 7: Push
```bash
git push -u origin main
```

## Alternative: All in one script

Create `push_to_github.bat`:
```batch
@echo off
cd F:\workspace\PVLDB\multi_agent_platform_evaluator

echo Initializing git...
git init

echo Adding files...
git add README.md .gitignore
git add agents/ platforms/ src/ config/ app.py requirements.txt
git add *.md
git add report/*.tex report/*.py report/*.md report/Makefile report/compile.bat

echo Committing...
git commit -m "Initial commit: Multi-agent platform evaluator framework"

echo Setting main branch...
git branch -M main

echo Adding remote...
git remote add origin https://github.com/mahdinaser/multi-agent-platform-evaluator.git

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause
```

## Notes

- Make sure you have Git installed and configured
- You may need to authenticate with GitHub (use GitHub CLI, SSH keys, or personal access token)
- Large files (experiments, data) are excluded via .gitignore
- PDF files in report/ are excluded - only source files are pushed

## If you get authentication errors:

1. **Use GitHub CLI:**
   ```bash
   gh auth login
   git push -u origin main
   ```

2. **Or use SSH:**
   ```bash
   git remote set-url origin git@github.com:mahdinaser/multi-agent-platform-evaluator.git
   git push -u origin main
   ```

3. **Or use Personal Access Token:**
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Create token with repo permissions
   - Use token as password when pushing

