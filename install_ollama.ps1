# PowerShell script to help install and setup Ollama

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Ollama Installation Helper" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is already installed
$ollamaInstalled = $false
try {
    $null = Get-Command ollama -ErrorAction Stop
    $ollamaInstalled = $true
    Write-Host "✓ Ollama is already installed!" -ForegroundColor Green
    ollama --version
} catch {
    Write-Host "✗ Ollama is not installed or not in PATH" -ForegroundColor Red
}

if (-not $ollamaInstalled) {
    Write-Host ""
    Write-Host "To install Ollama:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Download and Install (Recommended)" -ForegroundColor Cyan
    Write-Host "  1. Visit: https://ollama.ai/download" -ForegroundColor White
    Write-Host "  2. Download the Windows installer" -ForegroundColor White
    Write-Host "  3. Run the installer" -ForegroundColor White
    Write-Host "  4. Restart your terminal after installation" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 2: Using Winget (if available)" -ForegroundColor Cyan
    Write-Host "  winget install Ollama.Ollama" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 3: Using Chocolatey (if available)" -ForegroundColor Cyan
    Write-Host "  choco install ollama" -ForegroundColor White
    Write-Host ""
    
    # Try winget
    Write-Host "Attempting to install via Winget..." -ForegroundColor Yellow
    try {
        $wingetCheck = Get-Command winget -ErrorAction Stop
        Write-Host "Winget found. Installing Ollama..." -ForegroundColor Green
        winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
        Write-Host ""
        Write-Host "Installation complete! Please restart your terminal." -ForegroundColor Green
    } catch {
        Write-Host "Winget not available. Please use Option 1 (manual download)." -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "After installation, run this script again to verify." -ForegroundColor Cyan
    Write-Host ""
    pause
    exit
}

# Check if Ollama server is running
Write-Host ""
Write-Host "Checking Ollama server status..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ Ollama server is running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available models:" -ForegroundColor Cyan
    ollama list
} catch {
    Write-Host "✗ Ollama server is not running" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Starting Ollama server..." -ForegroundColor Cyan
    
    # Try to start as service
    try {
        $service = Get-Service -Name "*ollama*" -ErrorAction Stop
        if ($service.Status -ne 'Running') {
            Start-Service $service.Name
            Write-Host "✓ Started Ollama service" -ForegroundColor Green
            Start-Sleep -Seconds 3
        }
    } catch {
        # Service not found, try to start manually
        Write-Host "Starting Ollama server in background..." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
    }
    
    # Verify
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
        Write-Host "✓ Ollama server started successfully!" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Server may still be starting. Please wait a few seconds." -ForegroundColor Yellow
        Write-Host "  Or start manually: ollama serve" -ForegroundColor Yellow
    }
}

# Check if models are installed
Write-Host ""
Write-Host "Checking installed models..." -ForegroundColor Cyan
try {
    $models = ollama list
    if ($models -match "NAME") {
        Write-Host "✓ Models are installed" -ForegroundColor Green
    } else {
        Write-Host "⚠ No models installed" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To install a model, run:" -ForegroundColor Cyan
        Write-Host "  ollama pull llama2" -ForegroundColor White
        Write-Host ""
        Write-Host "Recommended models:" -ForegroundColor Cyan
        Write-Host "  ollama pull llama2      # General purpose (3.8GB)" -ForegroundColor White
        Write-Host "  ollama pull mistral     # Fast and efficient (4.1GB)" -ForegroundColor White
        Write-Host "  ollama pull codellama   # Code-focused (3.8GB)" -ForegroundColor White
    }
} catch {
    Write-Host "⚠ Could not check models. Server may not be running." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Your app is configured to use Ollama with model: llama2" -ForegroundColor Cyan
Write-Host "Make sure you have pulled the model: ollama pull llama2" -ForegroundColor Yellow
Write-Host ""

