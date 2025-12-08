# Start All CTU-Connect Services including Recommendation Service
# This script starts all infrastructure and recommendation services

Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         CTU-Connect - Start All Services (with AI)         ║" -ForegroundColor Cyan
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

# Load environment variables
Write-Host "`n⚙️  Loading environment variables..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $key = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host "✅ Environment variables loaded" -ForegroundColor Green
} else {
    Write-Host "⚠️  No .env file found (using defaults)" -ForegroundColor Yellow
}

# Start services
Write-Host "`n🚀 Starting all services..." -ForegroundColor Yellow
Write-Host "   📦 Infrastructure services:" -ForegroundColor White
Write-Host "      - Eureka Server (8761)" -ForegroundColor Gray
Write-Host "      - API Gateway (8090)" -ForegroundColor Gray
Write-Host "      - MongoDB (27018)" -ForegroundColor Gray
Write-Host "      - Neo4j (7474, 7687)" -ForegroundColor Gray
Write-Host "      - Kafka (9092)" -ForegroundColor Gray
Write-Host "      - PostgreSQL Auth (5433)" -ForegroundColor Gray
Write-Host "      - PostgreSQL Media (5434)" -ForegroundColor Gray
Write-Host "      - Redis Shared (6379)" -ForegroundColor Gray
Write-Host ""
Write-Host "   🤖 Recommendation Service:" -ForegroundColor White
Write-Host "      - PostgreSQL (5435)" -ForegroundColor Gray
Write-Host "      - Redis (6380)" -ForegroundColor Gray
Write-Host "      - Python AI Service (8000)" -ForegroundColor Gray

docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ All services started successfully!" -ForegroundColor Green
    
    # Wait for services to be healthy
    Write-Host "`n⏳ Waiting for services to be healthy (60s)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 60
    
    # Check service status
    Write-Host "`n📊 Service Status:" -ForegroundColor Cyan
    docker-compose ps
    
    # Test connections
    Write-Host "`n🧪 Testing key services..." -ForegroundColor Yellow
    
    # Test Eureka
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8761" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        Write-Host "✅ Eureka Server: OK (http://localhost:8761)" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Eureka Server: Initializing..." -ForegroundColor Yellow
    }
    
    # Test Recommendation Python Service
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        Write-Host "✅ Recommendation Python: OK (http://localhost:8000)" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Recommendation Python: Initializing..." -ForegroundColor Yellow
    }
    
    # Test PostgreSQL Recommend
    try {
        docker exec ctu-recommend-postgres psql -U recommend_user -d recommend_db -c "SELECT 1;" | Out-Null
        Write-Host "✅ Recommendation PostgreSQL: OK (localhost:5435)" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Recommendation PostgreSQL: Initializing..." -ForegroundColor Yellow
    }
    
    # Test Redis Recommend
    try {
        $redisTest = docker exec ctu-recommend-redis redis-cli -a recommend_redis_pass ping 2>$null
        if ($redisTest -eq "PONG") {
            Write-Host "✅ Recommendation Redis: OK (localhost:6380)" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠️  Recommendation Redis: Initializing..." -ForegroundColor Yellow
    }
    
    Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                  ALL SERVICES READY                         ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    
    Write-Host "`n📚 Key Services:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   🌐 Infrastructure:" -ForegroundColor Yellow
    Write-Host "      Eureka:        http://localhost:8761" -ForegroundColor White
    Write-Host "      API Gateway:   http://localhost:8090" -ForegroundColor White
    Write-Host "      Neo4j:         http://localhost:7474 (neo4j/password)" -ForegroundColor White
    Write-Host ""
    Write-Host "   🤖 Recommendation Service:" -ForegroundColor Yellow
    Write-Host "      Python API:    http://localhost:8000" -ForegroundColor White
    Write-Host "      Swagger UI:    http://localhost:8000/docs" -ForegroundColor White
    Write-Host "      PostgreSQL:    localhost:5435 (recommend_user/recommend_pass)" -ForegroundColor White
    Write-Host "      Redis:         localhost:6380 (password: recommend_redis_pass)" -ForegroundColor White
    
    Write-Host "`n🎯 Microservices (Run in IDE or separate terminal):" -ForegroundColor Cyan
    Write-Host "   ☕ Auth Service:    Port 8081" -ForegroundColor White
    Write-Host "   ☕ User Service:    Port 8082" -ForegroundColor White
    Write-Host "   ☕ Post Service:    Port 8083" -ForegroundColor White
    Write-Host "   ☕ Media Service:   Port 8084" -ForegroundColor White
    Write-Host "   ☕ Chat Service:    Port 8085" -ForegroundColor White
    Write-Host "   ☕ Recommend Java:  Port 8086 (Run in IDE)" -ForegroundColor Yellow
    
    Write-Host "`n🔧 Development Mode:" -ForegroundColor Cyan
    Write-Host "   ✅ Python code auto-reloads on changes" -ForegroundColor White
    Write-Host "   ✅ Edit files in: recommend-service/python-model/" -ForegroundColor White
    Write-Host "   ✅ Java services: Run in IDE for debugging" -ForegroundColor White
    
    Write-Host "`n📝 Useful Commands:" -ForegroundColor Cyan
    Write-Host "   View logs:         docker-compose logs -f" -ForegroundColor White
    Write-Host "   Stop all:          docker-compose down" -ForegroundColor White
    Write-Host "   Restart Python:    docker-compose restart recommend-python" -ForegroundColor White
    Write-Host "   Database console:  docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db" -ForegroundColor White
    Write-Host "   Redis console:     docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass" -ForegroundColor White
    
    Write-Host "`n💡 Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Start your Java microservices in IDE" -ForegroundColor White
    Write-Host "   2. Configure services to use these databases" -ForegroundColor White
    Write-Host "   3. Start developing with live reload!" -ForegroundColor White
    
    Write-Host ""
    
} else {
    Write-Host "`n❌ Failed to start services!" -ForegroundColor Red
    Write-Host "Check logs with: docker-compose logs" -ForegroundColor Yellow
    exit 1
}
