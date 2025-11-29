# Complete Ollama Setup Guide

## Quick Setup (Automated)

### Option 1: One-Click Setup
```powershell
cd PVLDB\multi_agent_platform_evaluator
.\auto_setup_ollama.ps1
```

This script will:
1. ✅ Check if Ollama is installed
2. ✅ Install Ollama if missing (via Winget)
3. ✅ Start Ollama server if not running
4. ✅ Pull llama2 model if not installed
5. ✅ Verify everything is working

### Option 2: Batch File (Windows)
```bash
setup_and_start_ollama.bat
```

### Option 3: Complete Setup (Ollama + App)
```powershell
.\complete_setup.ps1
```
This sets up Ollama AND runs the app!

## What the Scripts Do

### `auto_setup_ollama.ps1` (Recommended)
- **Simplest and most reliable**
- Installs Ollama via Winget
- Starts server automatically
- Pulls llama2 model
- Verifies everything

### `setup_and_start_ollama.ps1`
- **More detailed version**
- Better error messages
- More installation options
- Detailed status reporting

### `complete_setup.ps1`
- **All-in-one**
- Sets up Ollama
- Then runs app.py automatically

## Manual Setup (If Scripts Fail)

### 1. Install Ollama
```powershell
# Option A: Winget
winget install Ollama.Ollama

# Option B: Download
# Visit: https://ollama.ai/download
# Run installer
```

### 2. Restart Terminal
Close and reopen PowerShell/Command Prompt

### 3. Pull Model
```powershell
ollama pull llama2
```

### 4. Start Server
```powershell
ollama serve
```

### 5. Verify
```powershell
ollama list
```

## Troubleshooting

### Script Says "Ollama Not Found"
- Install manually from https://ollama.ai/download
- Restart terminal after installation
- Run script again

### Server Won't Start
- Check if already running: `curl http://localhost:11434/api/tags`
- Kill existing: `taskkill /F /IM ollama.exe`
- Start manually: `ollama serve`

### Model Pull Fails
- Check internet connection
- Model is ~3.8GB, may take time
- Try: `ollama pull llama2` manually

### Permission Errors
- Run PowerShell as Administrator
- Or install Ollama manually

## After Setup

Once setup is complete, you can:

1. **Run app with Ollama:**
   ```powershell
   python app.py
   ```

2. **Or use the batch file:**
   ```bash
   run_ollama_and_app.bat
   ```

The LLM agent will automatically use Ollama for real language model decisions!

## Verify Setup

Run this to check everything:
```powershell
# Check Ollama
ollama --version

# Check server
curl http://localhost:11434/api/tags

# Check models
ollama list
```

All should work if setup was successful!

