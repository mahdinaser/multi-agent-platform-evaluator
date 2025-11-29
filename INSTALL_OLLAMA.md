# Installing Ollama on Windows

## Quick Installation

### Method 1: Direct Download (Recommended)

1. **Visit:** https://ollama.ai/download
2. **Download:** Windows installer (OllamaSetup.exe)
3. **Run:** The installer
4. **Restart:** Your terminal/PowerShell
5. **Verify:**
   ```powershell
   ollama --version
   ```

### Method 2: Using Winget

```powershell
winget install Ollama.Ollama
```

### Method 3: Using Chocolatey

```powershell
choco install ollama
```

## After Installation

### 1. Restart Your Terminal
Close and reopen PowerShell/Command Prompt so PATH is updated.

### 2. Verify Installation
```powershell
ollama --version
```

### 3. Start Ollama Server
```powershell
ollama serve
```

Or it may start automatically as a Windows service.

### 4. Install a Model
```powershell
ollama pull llama2
```

### 5. Verify Setup
```powershell
ollama list
```

## Automated Setup Script

Run the helper script:
```powershell
cd PVLDB\multi_agent_platform_evaluator
.\install_ollama.ps1
```

This script will:
- Check if Ollama is installed
- Try to install it via Winget if available
- Start the Ollama server
- Check for installed models
- Provide next steps

## Troubleshooting

### "ollama is not recognized"
- **Solution:** Restart your terminal after installation
- **Or:** Add Ollama to PATH manually
  - Usually installed in: `%LOCALAPPDATA%\Programs\Ollama`
  - Add to PATH: `$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"`

### Server Won't Start
1. Check if already running:
   ```powershell
   Get-Process -Name ollama -ErrorAction SilentlyContinue
   ```

2. Check Windows service:
   ```powershell
   Get-Service -Name "*ollama*"
   ```

3. Start service:
   ```powershell
   Start-Service ollama
   ```

### Port Already in Use
If port 11434 is in use:
```powershell
netstat -ano | findstr :11434
taskkill /F /PID <process_id>
```

## Next Steps

Once Ollama is installed and running:

1. **Pull a model:**
   ```powershell
   ollama pull llama2
   ```

2. **Test the app:**
   ```powershell
   cd PVLDB\multi_agent_platform_evaluator
   python app.py
   ```

The LLM agent will automatically use Ollama if it's available!

