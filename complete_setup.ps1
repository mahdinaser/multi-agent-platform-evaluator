# Complete setup: Install Ollama, pull model, start server, and run app

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Complete Setup: Ollama + App" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Run Ollama setup first
Write-Host "Step 1: Setting up Ollama..." -ForegroundColor Yellow
& "$PSScriptRoot\setup_and_start_ollama.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Ollama setup had issues. Continuing anyway..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Step 2: Starting App" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Change to app directory
$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $appDir

# Check if Ollama server is running
$serverRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    $serverRunning = $true
    Write-Host "✓ Ollama server is running" -ForegroundColor Green
} catch {
    Write-Host "⚠ Ollama server not running. LLM agent will use fallback." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting app.py..." -ForegroundColor Cyan
Write-Host ""

# Run the app
python app.py

Write-Host ""
Write-Host "App finished. Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

