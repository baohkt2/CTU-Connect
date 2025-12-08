# Start Recommendation Service (Development Mode)
# This script starts PostgreSQL, Redis, and Python service with volume mount

Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Recommendation Service - Development Environment Startup   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Check if Docker is running
Write-Host "🔍 Checking Docker..." -ForegroundColor Yellow
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if network exists
Write-Host "`n🌐 Checking network..." -ForegroundColor Yellow
$networkExists = docker network ls | Select-String "ctu-connect-network"
if (-not $networkExists) {
    Write-Host "📡 Creating network ctu-connect-network..." -ForegroundColor Yellow
    docker network create ctu-connect-network
    Write-Host "✅ Network created" -ForegroundColor Green
} else {
    Write-Host "✅ Network already exists" -ForegroundColor Green
}

# Check if .env exists
Write-Host "`n⚙️  Checking configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✅ .env file found" -ForegroundColor Green
} else {
    if (Test-Path ".env.example") {
        Write-Host "📝 Creating .env from .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "✅ .env file created" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No .env file (using defaults)" -ForegroundColor Yellow
    }
}

# Start services
Write-Host "`n🚀 Starting services..." -ForegroundColor Yellow
Write-Host "   - PostgreSQL (Port 5433)" -ForegroundColor White
Write-Host "   - Redis (Port 6380)" -ForegroundColor White
Write-Host "   - Python Service (Port 8000 with live reload)" -ForegroundColor White

docker-compose -f docker-compose.dev.yml up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Services started successfully!" -ForegroundColor Green
    
    # Wait for services to be healthy
    Write-Host "`n⏳ Waiting for services to be healthy..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # Check service status
    Write-Host "`n📊 Service Status:" -ForegroundColor Cyan
    docker-compose -f docker-compose.dev.yml ps
    
    # Test connections
    Write-Host "`n🧪 Testing connections..." -ForegroundColor Yellow
    
    # Test Python service
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        Write-Host "✅ Python Service: OK (http://localhost:8000)" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Python Service: Waiting to start..." -ForegroundColor Yellow
    }
    
    # Test PostgreSQL
    try {
        docker exec ctu-recommend-postgres psql -U recommend_user -d recommend_db -c "SELECT 1;" | Out-Null
        Write-Host "✅ PostgreSQL: OK (localhost:5433)" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  PostgreSQL: Initializing..." -ForegroundColor Yellow
    }
    
    # Test Redis
    try {
        $redisTest = docker exec ctu-recommend-redis redis-cli -a recommend_redis_pass ping 2>$null
        if ($redisTest -eq "PONG") {
            Write-Host "✅ Redis: OK (localhost:6380)" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠️  Redis: Initializing..." -ForegroundColor Yellow
    }
    
    Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                    SERVICES READY                           ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    
    Write-Host "`n📚 Available Services:" -ForegroundColor Cyan
    Write-Host "   🐍 Python API:     http://localhost:8000" -ForegroundColor White
    Write-Host "   📖 Swagger UI:     http://localhost:8000/docs" -ForegroundColor White
    Write-Host "   🐘 PostgreSQL:     localhost:5433 (user: recommend_user, db: recommend_db)" -ForegroundColor White
    Write-Host "   🔴 Redis:          localhost:6380 (password: recommend_redis_pass)" -ForegroundColor White
    
    Write-Host "`n🔧 Development Mode:" -ForegroundColor Cyan
    Write-Host "   ✅ Python code auto-reloads on changes" -ForegroundColor White
    Write-Host "   ✅ Edit files in: recommend-service/python-model/" -ForegroundColor White
    Write-Host "   ✅ Java service: Run on IDE (Port 8081)" -ForegroundColor White
    
    Write-Host "`n📝 Useful Commands:" -ForegroundColor Cyan
    Write-Host "   View logs:         docker-compose -f docker-compose.dev.yml logs -f" -ForegroundColor White
    Write-Host "   Stop services:     docker-compose -f docker-compose.dev.yml down" -ForegroundColor White
    Write-Host "   Restart Python:    docker-compose -f docker-compose.dev.yml restart recommend-python" -ForegroundColor White
    Write-Host "   Database console:  docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db" -ForegroundColor White
    Write-Host "   Redis console:     docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass" -ForegroundColor White
    
    Write-Host "`n💡 Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Start your Java service in IDE (IntelliJ/Eclipse)" -ForegroundColor White
    Write-Host "   2. Configure Java to connect to:" -ForegroundColor White
    Write-Host "      - Python: http://localhost:8000" -ForegroundColor White
    Write-Host "      - PostgreSQL: localhost:5433" -ForegroundColor White
    Write-Host "      - Redis: localhost:6380" -ForegroundColor White
    Write-Host "   3. Start coding! Changes auto-reload." -ForegroundColor White
    
    Write-Host ""
    
} else {
    Write-Host "`n❌ Failed to start services!" -ForegroundColor Red
    Write-Host "Check logs with: docker-compose -f docker-compose.dev.yml logs" -ForegroundColor Yellow
    exit 1
}
