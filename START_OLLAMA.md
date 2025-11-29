# How to Start Ollama Server

## Quick Start

### Option 1: Automatic Start (Windows)

**Double-click:** `start_ollama.bat` or run:
```bash
start_ollama.bat
```

**Or PowerShell:**
```powershell
.\start_ollama.ps1
```

### Option 2: Manual Start

1. **Open a new terminal/command prompt**

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Keep that terminal open** (server runs in foreground)

4. **In another terminal, verify it's running:**
   ```bash
   ollama list
   ```

### Option 3: Background Service (Windows)

Ollama typically runs as a Windows service. Check if it's running:

```powershell
Get-Service -Name "*ollama*" -ErrorAction SilentlyContinue
```

If not running, start it:
```powershell
Start-Service ollama
```

## Verify Ollama is Running

### Test Connection:
```bash
curl http://localhost:11434/api/tags
```

Or use Python:
```bash
python test_ollama.py
```

### Check Available Models:
```bash
ollama list
```

## Install a Model (if needed)

If no models are installed:
```bash
ollama pull llama2
```

Other recommended models:
```bash
ollama pull mistral      # Fast and efficient
ollama pull codellama    # Code-focused
ollama pull llama2:13b   # Better quality (larger)
```

## Troubleshooting

### Ollama Not Found
- **Install Ollama:** Download from https://ollama.ai/
- **Add to PATH:** Restart terminal after installation
- **Windows:** Ollama should auto-add to PATH during installation

### Server Won't Start
1. Check if port 11434 is in use:
   ```powershell
   netstat -ano | findstr :11434
   ```

2. Kill existing Ollama processes:
   ```powershell
   taskkill /F /IM ollama.exe
   ```

3. Try starting again:
   ```bash
   ollama serve
   ```

### Connection Refused
- Make sure Ollama server is running
- Check firewall settings
- Verify port 11434 is accessible

## Running Ollama as Background Service

### Windows Service
Ollama installs as a Windows service. To manage it:

```powershell
# Check status
Get-Service ollama

# Start service
Start-Service ollama

# Stop service
Stop-Service ollama

# Restart service
Restart-Service ollama
```

### Manual Background Process
```powershell
# Start in background
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
```

## Verify for Your App

Once Ollama is running, test with:
```bash
cd PVLDB\multi_agent_platform_evaluator
python test_ollama.py
```

Or run your app - it will automatically use Ollama if available:
```bash
python app.py
```

## Default Configuration

The app is configured to use Ollama by default:
```yaml
llm_config:
  use_local: true
  use_ollama: true
  model_name: "llama2"
```

Make sure you have pulled the model:
```bash
ollama pull llama2
```

