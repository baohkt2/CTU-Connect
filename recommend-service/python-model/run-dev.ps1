# Script to run Python service in development mode
# Usage: .\run-dev.ps1

Write-Host "=== Starting Recommend Python Service (Dev Mode) ===" -ForegroundColor Green
Write-Host ""

# Set UTF-8 encoding for console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "✗ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup-venv.ps1 first:" -ForegroundColor Yellow
    Write-Host "  .\setup-venv.ps1" -ForegroundColor White
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠ .env file not found, using .env.example" -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

Write-Host "✓ Environment ready" -ForegroundColor Green
Write-Host ""

# Check if databases are running
Write-Host "Checking database connections..." -ForegroundColor Cyan
Write-Host "  - PostgreSQL: localhost:5435" -ForegroundColor Gray
Write-Host "  - Redis: localhost:6380" -ForegroundColor Gray
Write-Host ""
Write-Host "⚠ Make sure databases are running:" -ForegroundColor Yellow
Write-Host "   docker-compose up recommend-postgres recommend-redis neo4j kafka" -ForegroundColor Gray
Write-Host ""

# Start server
Write-Host "Starting FastAPI server on http://localhost:8000..." -ForegroundColor Cyan
Write-Host "  - API Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "  - Health Check: http://localhost:8000/health" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run uvicorn with reload (use --log-config for better encoding handling)
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload --log-level info
