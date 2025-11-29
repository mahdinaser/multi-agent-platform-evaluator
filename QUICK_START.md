# Quick Start Guide

## Run Everything at Once

### Option 1: Use the Batch File (Easiest)

**Double-click:** `run_ollama_and_app.bat`

This will:
1. Start Ollama server in one terminal window
2. Start app.py in another terminal window
3. Automatically handle everything

### Option 2: Manual Start (Two Terminals)

#### Terminal 1: Ollama Server
```bash
ollama serve
```
**Keep this terminal open!**

#### Terminal 2: Run App
```bash
cd PVLDB\multi_agent_platform_evaluator
python app.py
```

## Before Running

### 1. Install Ollama (if not installed)
```bash
# Download from: https://ollama.ai/download
# Or use winget:
winget install Ollama.Ollama
```

### 2. Install Python Dependencies
```bash
cd PVLDB\multi_agent_platform_evaluator
pip install -r requirements.txt
```

### 3. Pull a Model (if needed)
```bash
ollama pull llama2
```

## Verify Setup

### Check Ollama:
```bash
ollama --version
ollama list
```

### Check Python:
```bash
python --version
python -c "import ollama; print('Ollama package OK')"
```

## Troubleshooting

### Ollama Not Found
- Install from: https://ollama.ai/download
- Restart terminal after installation

### Server Won't Start
- Check if already running: `curl http://localhost:11434/api/tags`
- Kill existing: `taskkill /F /IM ollama.exe`
- Start again: `ollama serve`

### App Can't Connect to Ollama
- Make sure Ollama server is running in Terminal 1
- Wait a few seconds after starting server
- Check: `curl http://localhost:11434/api/tags`

## What Happens

1. **Ollama Server Terminal:**
   - Runs `ollama serve`
   - Serves LLM models on port 11434
   - Must stay open while app runs

2. **App Terminal:**
   - Runs `python app.py`
   - Generates data
   - Runs experiments
   - Uses Ollama for LLM agent decisions
   - Generates analysis and plots

## Expected Output

### Ollama Terminal:
```
Ollama Server Terminal
========================================
Keep this window open!
========================================
ollama serve
```

### App Terminal:
```
================================================
Research Experiment Engine - Multi-Agent Multi-Platform Evaluation
PVLDB Paper Framework
================================================

Step 1: Creating folder structure...
Step 2: Generating data sources...
Step 3: Initializing experiment runner...
Step 4: Running comprehensive experiments...
...
```

## Stop Everything

1. Close the App Terminal (Ctrl+C if running)
2. Close the Ollama Server Terminal (Ctrl+C)
3. Or run: `taskkill /F /IM ollama.exe`

