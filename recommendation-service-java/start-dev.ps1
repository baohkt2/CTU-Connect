# Start Development Environment
# Databases in Docker, Service in IDE

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Recommendation Service - Dev Setup" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Check Docker
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting databases..." -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml up -d

Write-Host ""
Write-Host "Waiting for databases to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "Checking database status..." -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml ps

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "✅ Databases are ready!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Database Endpoints:" -ForegroundColor Cyan
Write-Host "  PostgreSQL: localhost:5435" -ForegroundColor White
Write-Host "  Neo4j HTTP: http://localhost:7474" -ForegroundColor White
Write-Host "  Neo4j Bolt: bolt://localhost:7687" -ForegroundColor White
Write-Host "  Redis:      localhost:6379" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Open project in IntelliJ IDEA or VS Code" -ForegroundColor White
Write-Host "  2. Run with profile: dev" -ForegroundColor White
Write-Host "  3. Or run: mvn spring-boot:run -Dspring-boot.run.profiles=dev" -ForegroundColor White
Write-Host ""
Write-Host "To stop databases:" -ForegroundColor Yellow
Write-Host "  docker-compose -f docker-compose.dev.yml down" -ForegroundColor White
Write-Host ""
