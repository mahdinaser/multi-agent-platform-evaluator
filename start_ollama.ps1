# PowerShell script to start Ollama server

Write-Host "Starting Ollama server..." -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is installed
try {
    $ollamaVersion = ollama --version 2>&1
    Write-Host "Ollama found: $ollamaVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Ollama is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Ollama from: https://ollama.ai/" -ForegroundColor Yellow
    Write-Host "After installation, add Ollama to your PATH or restart your terminal" -ForegroundColor Yellow
    exit 1
}

# Check if Ollama is already running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "Ollama server is already running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available models:" -ForegroundColor Cyan
    ollama list
    exit 0
} catch {
    Write-Host "Ollama server not running, starting it..." -ForegroundColor Yellow
}

# Start Ollama server
Write-Host "Starting Ollama server in background..." -ForegroundColor Cyan
$process = Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -PassThru

# Wait for server to start
Write-Host "Waiting for server to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if server started
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host ""
    Write-Host "Ollama server started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Server is running on http://localhost:11434" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Available models:" -ForegroundColor Cyan
    ollama list
    Write-Host ""
    Write-Host "To stop the server, run: Stop-Process -Id $($process.Id)" -ForegroundColor Yellow
    Write-Host "Or close this PowerShell window" -ForegroundColor Yellow
} catch {
    Write-Host ""
    Write-Host "WARNING: Could not verify Ollama server is running" -ForegroundColor Red
    Write-Host "Please check if Ollama is installed correctly" -ForegroundColor Yellow
    Write-Host "Process ID: $($process.Id)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

