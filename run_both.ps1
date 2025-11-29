# PowerShell script to run Ollama server and app.py in separate terminals

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Starting Ollama Server and App in Separate Terminals" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Get current directory
$appDir = "F:\workspace\PVLDB\multi_agent_platform_evaluator"

# Check if Ollama is installed
Write-Host "Checking Ollama installation..." -ForegroundColor Yellow
try {
    $null = Get-Command ollama -ErrorAction Stop
    Write-Host "✓ Ollama is installed" -ForegroundColor Green
} catch {
    Write-Host "✗ Ollama is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Ollama first:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://ollama.ai/download" -ForegroundColor White
    Write-Host "  2. Install and restart terminal" -ForegroundColor White
    Write-Host "  3. Run this script again" -ForegroundColor White
    Write-Host ""
    pause
    exit 1
}

# Check if Ollama server is already running
Write-Host ""
Write-Host "Checking if Ollama server is running..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ Ollama server is already running" -ForegroundColor Green
    $startNewServer = $false
} catch {
    Write-Host "✗ Ollama server is not running" -ForegroundColor Yellow
    $startNewServer = $true
}

# Start Ollama server in new terminal
if ($startNewServer) {
    Write-Host ""
    Write-Host "Starting Ollama server in new terminal..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "`$Host.UI.RawUI.WindowTitle = 'Ollama Server'; " +
        "Write-Host '========================================' -ForegroundColor Cyan; " +
        "Write-Host 'Ollama Server Terminal' -ForegroundColor Cyan; " +
        "Write-Host '========================================' -ForegroundColor Cyan; " +
        "Write-Host ''; " +
        "Write-Host 'Starting Ollama server...' -ForegroundColor Yellow; " +
        "Write-Host 'Server will run on http://localhost:11434' -ForegroundColor Gray; " +
        "Write-Host 'Keep this window open!' -ForegroundColor Yellow; " +
        "Write-Host ''; " +
        "ollama serve; " +
        "Write-Host ''; " +
        "Write-Host 'Server stopped. Press any key to exit...' -ForegroundColor Red; " +
        "`$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')"
    )
    Write-Host "✓ Ollama server terminal opened" -ForegroundColor Green
    Write-Host "  Waiting for server to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Verify server started
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
        Write-Host "✓ Ollama server is running!" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Server may still be starting. Please wait..." -ForegroundColor Yellow
    }
} else {
    Write-Host "  Using existing Ollama server" -ForegroundColor Gray
}

# Check for models
Write-Host ""
Write-Host "Checking installed models..." -ForegroundColor Yellow
try {
    $models = ollama list 2>&1
    if ($models -match "NAME" -or $models -match "llama") {
        Write-Host "✓ Models are installed" -ForegroundColor Green
    } else {
        Write-Host "⚠ No models found. Installing llama2..." -ForegroundColor Yellow
        Write-Host "  This may take a few minutes..." -ForegroundColor Gray
        ollama pull llama2
    }
} catch {
    Write-Host "⚠ Could not check models" -ForegroundColor Yellow
}

# Start app.py in new terminal
Write-Host ""
Write-Host "Starting app.py in new terminal..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$appDir'; " +
    "`$Host.UI.RawUI.WindowTitle = 'Research Experiment Engine'; " +
    "Write-Host '========================================' -ForegroundColor Green; " +
    "Write-Host 'Research Experiment Engine' -ForegroundColor Green; " +
    "Write-Host 'Multi-Agent Multi-Platform Evaluation' -ForegroundColor Green; " +
    "Write-Host '========================================' -ForegroundColor Green; " +
    "Write-Host ''; " +
    "python app.py; " +
    "Write-Host ''; " +
    "Write-Host 'App finished. Press any key to exit...' -ForegroundColor Yellow; " +
    "`$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')"
)

Write-Host "✓ App terminal opened" -ForegroundColor Green
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Two terminals are now open:" -ForegroundColor Cyan
Write-Host "  1. Ollama Server Terminal - Keep this open!" -ForegroundColor Yellow
Write-Host "  2. App Terminal - Running your experiments" -ForegroundColor Green
Write-Host ""
Write-Host "The app will automatically use Ollama for LLM agent decisions." -ForegroundColor Gray
Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

