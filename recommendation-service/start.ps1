# CTU Connect Recommendation Service Startup Script (PowerShell version, no emoji)

try {
    Write-Host "Starting CTU Connect Recommendation Service..."

    # Check if Python is installed
    if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) {
        Write-Error "Python 3 is required but not installed."
        exit 1
    }

    # Check if Docker is installed
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is required but not installed."
        exit 1
    }

    if (-not (Test-Path -Path ".\venv")) {
        Write-Host "Creating virtual environment..."
        python3 -m venv venv
    }

    Write-Host "Activating virtual environment..."
    . .\venv\Scripts\Activate.ps1

    Write-Host "Installing dependencies..."
    pip install -r requirements.txt

    if (-not (Test-Path -Path ".env")) {
        Write-Host "Creating environment configuration..."
        $envContent = @"
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5433/recommendation_db

# Redis Configuration
REDIS_URL=redis://localhost:6380

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Service Configuration
SECRET_KEY=your-secret-key-here
DEBUG=true
HOST=0.0.0.0
PORT=8000

# ML Configuration
MLFLOW_TRACKING_URI=http://localhost:5001
MODEL_PATH=./models/recommendation_model.pt

# Recommendation Settings
TOP_K_RECOMMENDATIONS=10
EMBEDDING_DIM=256
NUM_HEADS=8
"@
        $envContent | Set-Content -Path ".env"
        Write-Host "Created .env file. Please review and update the configuration."
    }

    Write-Host "Loading environment variables..."
    Get-Content .env | ForEach-Object {
        if ($_ -notmatch '^#' -and $_ -match '=') {
            $parts = $_ -split '=', 2
            Set-Item -Path "Env:$($parts[0])" -Value $parts[1]
        }
    }

    Write-Host "Starting supporting services (Database, Redis, MLflow)..."
    docker-compose up -d recommendation-db recommendation-redis mlflow

    Write-Host "Waiting for services to be ready..."
    Start-Sleep -Seconds 10

    Write-Host "Checking database connection..."
    $dbCheckScript = @"
import asyncio
import asyncpg
import sys

async def check_db():
    try:
        conn = await asyncpg.connect('postgresql://postgres:password@localhost:5433/recommendation_db')
        await conn.close()
        print('Database connection successful')
        return True
    except Exception as e:
        print(f'Database connection failed: {e}')
        return False

if not asyncio.run(check_db()):
    sys.exit(1)
"@
    python3 -c $dbCheckScript
    if ($LASTEXITCODE -ne 0) { exit 1 }

    Write-Host "Initializing database tables..."
    $initDbScript = @"
import asyncio
from db.models import create_tables

async def init_db():
    try:
        await create_tables()
        print('Database tables created successfully')
    except Exception as e:
        print(f'Database initialization failed: {e}')
        raise

asyncio.run(init_db())
"@
    python3 -c $initDbScript

    if (-not (Test-Path -Path ".\models")) {
        New-Item -ItemType Directory -Path ".\models" | Out-Null
    }

    $port = $env:PORT
    if (-not $port) { $port = 8000 }
    Write-Host "Starting recommendation service..."
    Write-Host "Service will be available at: http://localhost:$port"
    Write-Host "MLflow UI available at: http://localhost:5001"
    Write-Host "API Documentation available at: http://localhost:$port/docs"
    Write-Host ""
    Write-Host "To stop the service, press Ctrl+C"
    Write-Host "To stop all services, run: docker-compose down"
    Write-Host ""

    python3 main.py

} catch {
    Write-Error "An error occurred: $_"
    exit 1
}
