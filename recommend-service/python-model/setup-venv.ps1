# Script to setup Python virtual environment for Recommend Service
# Usage: .\setup-venv.ps1

Write-Host "=== CTU Connect - Recommend Service Python Setup ===" -ForegroundColor Green
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
    
    # Check if Python 3.10 or higher
    if ($pythonVersion -match "Python 3\.([0-9]+)\.") {
        $minorVersion = [int]$Matches[1]
        if ($minorVersion -lt 10) {
            Write-Host "⚠ Warning: Python 3.10+ recommended, you have $pythonVersion" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "✗ Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "⚠ Virtual environment already exists. Delete it? (y/n): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq 'y') {
        Remove-Item -Recurse -Force venv
        Write-Host "✓ Removed old virtual environment" -ForegroundColor Green
    } else {
        Write-Host "Skipping venv creation..." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To activate existing venv, run:" -ForegroundColor Cyan
        Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
        exit 0
    }
}

python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host ""

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
Write-Host "⚠ This may take several minutes (PyTorch is large)..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Check .env file
Write-Host "Checking configuration files..." -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✓ Created .env file from .env.example" -ForegroundColor Green
    Write-Host "⚠ Please review and update .env file if needed" -ForegroundColor Yellow
} else {
    Write-Host "✓ .env file exists" -ForegroundColor Green
}

Write-Host ""

# Display next steps
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start databases with Docker:" -ForegroundColor White
Write-Host "   docker-compose up postgres recommend-redis neo4j kafka" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Run the Python service:" -ForegroundColor White
Write-Host "   uvicorn server:app --host 0.0.0.0 --port 8000 --reload" -ForegroundColor Gray
Write-Host "   OR" -ForegroundColor Gray
Write-Host "   python server.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. API will be available at:" -ForegroundColor White
Write-Host "   - API: http://localhost:8000" -ForegroundColor Gray
Write-Host "   - Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "   - Health: http://localhost:8000/health" -ForegroundColor Gray
Write-Host ""
Write-Host "To activate venv later, run:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
