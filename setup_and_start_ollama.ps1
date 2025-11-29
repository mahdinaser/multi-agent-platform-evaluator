# PowerShell script to install Ollama, pull model, and start server
# Comprehensive setup script

$ErrorActionPreference = "Continue"
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Ollama Complete Setup and Start Script" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Ollama is installed
Write-Host "[Step 1/4] Checking Ollama installation..." -ForegroundColor Yellow
$ollamaInstalled = $false
$ollamaPath = $null

try {
    $ollamaVersion = ollama --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $ollamaInstalled = $true
        Write-Host "✓ Ollama is already installed: $ollamaVersion" -ForegroundColor Green
    }
} catch {
    # Check common installation paths
    $commonPaths = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "C:\Program Files\Ollama\ollama.exe",
        "C:\Program Files (x86)\Ollama\ollama.exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $ollamaInstalled = $true
            $ollamaPath = $path
            $env:PATH += ";$(Split-Path $path -Parent)"
            Write-Host "✓ Found Ollama at: $path" -ForegroundColor Green
            break
        }
    }
}

# Step 2: Install Ollama if not installed
if (-not $ollamaInstalled) {
    Write-Host ""
    Write-Host "[Step 2/4] Installing Ollama..." -ForegroundColor Yellow
    
    # Try Winget first
    $installed = $false
    try {
        $wingetCheck = Get-Command winget -ErrorAction Stop
        Write-Host "  Attempting installation via Winget..." -ForegroundColor Cyan
        $installOutput = winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Ollama installed via Winget" -ForegroundColor Green
            $installed = $true
            # Refresh PATH
            $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        }
    } catch {
        Write-Host "  Winget not available" -ForegroundColor Gray
    }
    
    # Try Chocolatey
    if (-not $installed) {
        try {
            $chocoCheck = Get-Command choco -ErrorAction Stop
            Write-Host "  Attempting installation via Chocolatey..." -ForegroundColor Cyan
            choco install ollama -y
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ Ollama installed via Chocolatey" -ForegroundColor Green
                $installed = $true
            }
        } catch {
            Write-Host "  Chocolatey not available" -ForegroundColor Gray
        }
    }
    
    if (-not $installed) {
        Write-Host ""
        Write-Host "✗ Automatic installation failed" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install Ollama manually:" -ForegroundColor Yellow
        Write-Host "  1. Download from: https://ollama.ai/download" -ForegroundColor White
        Write-Host "  2. Run the installer (OllamaSetup.exe)" -ForegroundColor White
        Write-Host "  3. Restart your terminal" -ForegroundColor White
        Write-Host "  4. Run this script again" -ForegroundColor White
        Write-Host ""
        Write-Host "Opening download page..." -ForegroundColor Cyan
        Start-Process "https://ollama.ai/download"
        pause
        exit 1
    }
    
    # Wait a bit for installation to complete
    Write-Host "  Waiting for installation to complete..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
    
    # Refresh PATH and verify
    $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    try {
        $null = Get-Command ollama -ErrorAction Stop
        Write-Host "✓ Ollama installation verified" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Ollama installed but not in PATH. Please restart terminal." -ForegroundColor Yellow
        Write-Host "  Or add manually: $env:LOCALAPPDATA\Programs\Ollama" -ForegroundColor Gray
    }
}

# Step 3: Check if server is running
Write-Host ""
Write-Host "[Step 3/4] Checking Ollama server status..." -ForegroundColor Yellow
$serverRunning = $false

try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    $serverRunning = $true
    Write-Host "✓ Ollama server is already running" -ForegroundColor Green
} catch {
    Write-Host "✗ Ollama server is not running" -ForegroundColor Yellow
    
    # Try to start as Windows service
    try {
        $service = Get-Service -Name "*ollama*" -ErrorAction Stop
        if ($service.Status -ne 'Running') {
            Write-Host "  Starting Ollama service..." -ForegroundColor Cyan
            Start-Service $service.Name
            Start-Sleep -Seconds 3
            if ($service.Status -eq 'Running') {
                $serverRunning = $true
                Write-Host "✓ Ollama service started" -ForegroundColor Green
            }
        } else {
            $serverRunning = $true
        }
    } catch {
        # Service not found, start manually
        Write-Host "  Starting Ollama server manually..." -ForegroundColor Cyan
        try {
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 5
            
            # Verify
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
                $serverRunning = $true
                Write-Host "✓ Ollama server started successfully" -ForegroundColor Green
            } catch {
                Write-Host "⚠ Server may still be starting. Will check again..." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "✗ Failed to start server: $_" -ForegroundColor Red
        }
    }
}

# Final check for server
if (-not $serverRunning) {
    Write-Host ""
    Write-Host "Attempting final server start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
        $serverRunning = $true
        Write-Host "✓ Server is now running" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Server may need manual start. Run: ollama serve" -ForegroundColor Yellow
    }
}

# Step 4: Check and pull model
Write-Host ""
Write-Host "[Step 4/4] Checking and installing model (llama2)..." -ForegroundColor Yellow

if ($serverRunning) {
    try {
        # Get list of installed models
        $modelsOutput = ollama list 2>&1
        $hasLlama2 = $modelsOutput -match "llama2"
        
        if ($hasLlama2) {
            Write-Host "✓ Model 'llama2' is already installed" -ForegroundColor Green
        } else {
            Write-Host "  Model 'llama2' not found. Installing..." -ForegroundColor Cyan
            Write-Host "  This may take several minutes (model size: ~3.8GB)..." -ForegroundColor Gray
            Write-Host ""
            
            # Pull the model
            $pullProcess = Start-Process -FilePath "ollama" -ArgumentList "pull", "llama2" -NoNewWindow -Wait -PassThru
            
            if ($pullProcess.ExitCode -eq 0) {
                Write-Host "✓ Model 'llama2' installed successfully" -ForegroundColor Green
            } else {
                Write-Host "✗ Model installation failed. You can install manually: ollama pull llama2" -ForegroundColor Red
            }
        }
        
        # Show installed models
        Write-Host ""
        Write-Host "Installed models:" -ForegroundColor Cyan
        ollama list
        
    } catch {
        Write-Host "⚠ Could not check/install models: $_" -ForegroundColor Yellow
        Write-Host "  You can install manually: ollama pull llama2" -ForegroundColor Gray
    }
} else {
    Write-Host "⚠ Server not running. Cannot check/install models." -ForegroundColor Yellow
    Write-Host "  Please start server first: ollama serve" -ForegroundColor Gray
}

# Summary
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Setup Summary" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

if ($ollamaInstalled -or (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "✓ Ollama: Installed" -ForegroundColor Green
} else {
    Write-Host "✗ Ollama: Not installed" -ForegroundColor Red
}

if ($serverRunning) {
    Write-Host "✓ Server: Running on http://localhost:11434" -ForegroundColor Green
} else {
    Write-Host "✗ Server: Not running" -ForegroundColor Red
    Write-Host "  Start manually: ollama serve" -ForegroundColor Yellow
}

try {
    $modelsOutput = ollama list 2>&1
    if ($modelsOutput -match "llama2") {
        Write-Host "✓ Model: llama2 installed" -ForegroundColor Green
    } else {
        Write-Host "⚠ Model: llama2 not installed" -ForegroundColor Yellow
        Write-Host "  Install: ollama pull llama2" -ForegroundColor Gray
    }
} catch {
    Write-Host "⚠ Model: Cannot verify" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. If server is running, you can now run: python app.py" -ForegroundColor White
Write-Host "  2. To start server manually: ollama serve" -ForegroundColor White
Write-Host "  3. To install more models: ollama pull <model-name>" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

