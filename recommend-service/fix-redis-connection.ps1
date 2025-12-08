# Script to diagnose and fix Redis connection issues

Write-Host "`n=== Redis Connection Diagnostic ===" -ForegroundColor Green
Write-Host ""

# Check Docker containers
Write-Host "1. Checking Docker containers..." -ForegroundColor Cyan
$containers = docker ps --filter "name=redis" --format "{{.Names}}"
Write-Host "   Redis containers running: $($containers -join ', ')" -ForegroundColor Gray

# Check if recommend-redis is running
if ($containers -notcontains "ctu-recommend-redis") {
    Write-Host "   ✗ recommend-redis container not running!" -ForegroundColor Red
    Write-Host ""
    Write-Host "   Starting recommend-redis..." -ForegroundColor Yellow
    docker-compose up -d recommend-redis
    Start-Sleep -Seconds 5
}

# Test connections
Write-Host "`n2. Testing Redis connections..." -ForegroundColor Cyan

# Test recommend-redis with password
Write-Host "   Testing recommend-redis (6380)..." -ForegroundColor Gray
try {
    $result = docker exec ctu-recommend-redis redis-cli -a recommend_redis_pass ping 2>$null
    if ($result -eq "PONG") {
        Write-Host "   ✓ recommend-redis is responding" -ForegroundColor Green
    }
} catch {
    Write-Host "   ✗ recommend-redis not responding" -ForegroundColor Red
}

# Check port mapping
Write-Host "`n3. Checking port mappings..." -ForegroundColor Cyan
$port6380 = netstat -ano | findstr ":6380" | Select-Object -First 1
if ($port6380) {
    Write-Host "   ✓ Port 6380 is bound" -ForegroundColor Green
    Write-Host "     $port6380" -ForegroundColor Gray
} else {
    Write-Host "   ✗ Port 6380 not bound" -ForegroundColor Red
}

# Check Java application.yml config
Write-Host "`n4. Checking application-dev.yml config..." -ForegroundColor Cyan
$configPath = "..\recommend-service\java-api\src\main\resources\application-dev.yml"
if (Test-Path $configPath) {
    $redisConfig = Select-String -Path $configPath -Pattern "redis" -Context 0,3
    if ($redisConfig) {
        Write-Host "   Redis configuration found:" -ForegroundColor Gray
        $redisConfig | ForEach-Object { Write-Host "     $($_.Line)" -ForegroundColor Gray }
    }
}

# Test connection from host
Write-Host "`n5. Testing Redis connection from host..." -ForegroundColor Cyan
try {
    # Using redis-cli if available, or telnet
    $testCommand = "docker run --rm --network ctu-connect-demo_ctuconnect-network redis:7-alpine redis-cli -h recommend-redis -p 6379 -a recommend_redis_pass ping"
    $result = Invoke-Expression $testCommand 2>&1
    if ($result -like "*PONG*") {
        Write-Host "   ✓ Can connect to recommend-redis from Docker network" -ForegroundColor Green
    } else {
        Write-Host "   ✗ Cannot connect to recommend-redis" -ForegroundColor Red
        Write-Host "     Result: $result" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠ Could not test Docker network connection" -ForegroundColor Yellow
}

Write-Host "`n6. Recommendations:" -ForegroundColor Cyan
Write-Host "   • Make sure recommend-redis container is running" -ForegroundColor White
Write-Host "   • Verify application-dev.yml has correct config:" -ForegroundColor White
Write-Host "     - host: localhost" -ForegroundColor Gray
Write-Host "     - port: 6380" -ForegroundColor Gray
Write-Host "     - password: recommend_redis_pass" -ForegroundColor Gray
Write-Host "   • Restart Java service after config changes" -ForegroundColor White
Write-Host ""
