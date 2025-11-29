# Simplified automatic Ollama setup script
# Installs, pulls model, and starts server

param(
    [string]$ModelName = "llama2"
)

$ErrorActionPreference = "Continue"

function Write-Step($step, $message) {
    Write-Host "[$step] $message" -ForegroundColor Yellow
}

function Write-Success($message) {
    Write-Host "✓ $message" -ForegroundColor Green
}

function Write-Error($message) {
    Write-Host "✗ $message" -ForegroundColor Red
}

function Write-Info($message) {
    Write-Host "  $message" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ollama Auto Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check/Install Ollama
Write-Step "1/4" "Checking Ollama installation..."

$ollamaCmd = $null
try {
    $ollamaCmd = Get-Command ollama -ErrorAction Stop
    Write-Success "Ollama is installed: $(ollama --version 2>&1 | Select-Object -First 1)"
} catch {
    Write-Info "Ollama not found. Attempting installation..."
    
    # Try Winget
    try {
        $null = Get-Command winget -ErrorAction Stop
        Write-Info "Installing via Winget..."
        $result = winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Ollama installed via Winget"
            Start-Sleep -Seconds 3
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        } else {
            throw "Winget installation failed"
        }
    } catch {
        Write-Error "Automatic installation failed"
        Write-Host ""
        Write-Host "Please install Ollama manually:" -ForegroundColor Yellow
        Write-Host "  1. Download: https://ollama.ai/download" -ForegroundColor White
        Write-Host "  2. Install the application" -ForegroundColor White
        Write-Host "  3. Restart terminal and run this script again" -ForegroundColor White
        Write-Host ""
        Start-Process "https://ollama.ai/download"
        exit 1
    }
}

# Step 2: Check/Start Server
Write-Host ""
Write-Step "2/4" "Checking Ollama server..."

$serverRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    $serverRunning = $true
    Write-Success "Server is already running"
} catch {
    Write-Info "Server not running. Starting..."
    
    # Try service first
    try {
        $service = Get-Service -Name "*ollama*" -ErrorAction Stop
        if ($service.Status -ne 'Running') {
            Start-Service $service.Name
            Start-Sleep -Seconds 2
        }
        $serverRunning = $true
        Write-Success "Server started via service"
    } catch {
        # Start manually
        try {
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 5
            
            # Verify
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
                $serverRunning = $true
                Write-Success "Server started successfully"
            } catch {
                Write-Error "Server may still be starting. Please wait..."
            }
        } catch {
            Write-Error "Failed to start server: $_"
            Write-Info "Start manually: ollama serve"
        }
    }
}

# Step 3: Check/Pull Model
Write-Host ""
Write-Step "3/4" "Checking model '$ModelName'..."

if ($serverRunning) {
    try {
        $models = ollama list 2>&1
        if ($models -match $ModelName) {
            Write-Success "Model '$ModelName' is already installed"
        } else {
            Write-Info "Model not found. Installing (this may take several minutes)..."
            Write-Info "Model size: ~3.8GB for llama2"
            Write-Host ""
            
            ollama pull $ModelName
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Model '$ModelName' installed successfully"
            } else {
                Write-Error "Model installation failed"
                Write-Info "Install manually: ollama pull $ModelName"
            }
        }
    } catch {
        Write-Error "Could not check/install models: $_"
    }
} else {
    Write-Error "Server not running. Cannot install models."
    Write-Info "Start server first: ollama serve"
}

# Step 4: Final Verification
Write-Host ""
Write-Step "4/4" "Final verification..."

$allGood = $true

# Check Ollama
try {
    $null = Get-Command ollama -ErrorAction Stop
    Write-Success "Ollama: Installed"
} catch {
    Write-Error "Ollama: Not found"
    $allGood = $false
}

# Check Server
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Success "Server: Running on http://localhost:11434"
} catch {
    Write-Error "Server: Not running"
    Write-Info "Start manually: ollama serve"
    $allGood = $false
}

# Check Model
try {
    $models = ollama list 2>&1
    if ($models -match $ModelName) {
        Write-Success "Model: $ModelName installed"
    } else {
        Write-Error "Model: $ModelName not installed"
        Write-Info "Install: ollama pull $ModelName"
        $allGood = $false
    }
} catch {
    Write-Error "Model: Cannot verify"
    $allGood = $false
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "✓ Setup Complete! Ollama is ready to use." -ForegroundColor Green
} else {
    Write-Host "⚠ Setup completed with some issues." -ForegroundColor Yellow
    Write-Host "  Check the messages above for details." -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($allGood) {
    Write-Host "You can now:" -ForegroundColor Cyan
    Write-Host "  - Run: python app.py" -ForegroundColor White
    Write-Host "  - Or: run_ollama_and_app.bat" -ForegroundColor White
} else {
    Write-Host "To complete setup:" -ForegroundColor Yellow
    Write-Host "  1. Make sure Ollama is installed" -ForegroundColor White
    Write-Host "  2. Start server: ollama serve" -ForegroundColor White
    Write-Host "  3. Install model: ollama pull $ModelName" -ForegroundColor White
}

Write-Host ""

