# Simple API Test Script for Recommend Service
# Quick health checks and basic functionality tests

Write-Host "`n=== Recommend Service Simple Test ===" -ForegroundColor Green
Write-Host ""

$pythonUrl = "http://localhost:8000"
$javaUrl = "http://localhost:8095"

# Test Python Service
Write-Host "Testing Python ML Service..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$pythonUrl/health" -Method GET -TimeoutSec 5
    Write-Host "✓ Python service is running" -ForegroundColor Green
    Write-Host "  Status: $($response.status)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Python service is not running on port 8000" -ForegroundColor Red
    Write-Host "  Start it with: .\python-model\run-dev.ps1" -ForegroundColor Yellow
}

Write-Host ""

# Test Java Service - Try API endpoint first, then health
Write-Host "Testing Java Recommendation Service..." -ForegroundColor Cyan
$javaRunning = $false
try {
    # Try actual API endpoint first (more reliable)
    $response = Invoke-RestMethod -Uri "$javaUrl/api/recommendation/feed?userId=health_check&size=1" -Method GET -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ Java service is running" -ForegroundColor Green
    Write-Host "  API endpoint responding" -ForegroundColor Gray
    $javaRunning = $true
    
    # Check health status separately
    try {
        $health = Invoke-RestMethod -Uri "$javaUrl/actuator/health" -Method GET -TimeoutSec 5
        if ($health.status -eq "UP") {
            Write-Host "  Health status: UP" -ForegroundColor Green
        } else {
            Write-Host "  Health status: $($health.status) (but API works)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  Health endpoint: DOWN (but API works)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Java service is not running on port 8095" -ForegroundColor Red
    Write-Host "  Start it in IntelliJ IDEA with profile: dev" -ForegroundColor Yellow
}

Write-Host ""

# Test embedding functionality
Write-Host "Testing embedding API..." -ForegroundColor Cyan
try {
    $body = @{
        post_id = "test_001"
        content = "Thông báo học bổng VIED 2024"
        tags = @("scholarship")
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$pythonUrl/embed/post" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
    Write-Host "✓ Embedding API works" -ForegroundColor Green
    Write-Host "  Embedding dimension: $($response.embedding.Count)" -ForegroundColor Gray
} catch {
    Write-Host "✗ Embedding API failed" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""

# Test recommendation API (only if Java is running)
if ($javaRunning) {
    Write-Host "Testing recommendation API..." -ForegroundColor Cyan
    try {
        $response = Invoke-RestMethod -Uri "$javaUrl/api/recommendation/feed?userId=user_test&size=5" -Method GET -TimeoutSec 10
        Write-Host "✓ Recommendation API works" -ForegroundColor Green
        Write-Host "  Recommendations returned: $($response.recommendations.Count)" -ForegroundColor Gray
    } catch {
        Write-Host "✗ Recommendation API failed" -ForegroundColor Red
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Test Summary ===" -ForegroundColor Cyan
Write-Host "If you see health status DOWN but API works, this is usually due to:" -ForegroundColor Gray
Write-Host "  - Redis connection issue (check: docker ps | findstr redis)" -ForegroundColor Gray
Write-Host "  - Eureka not configured (acceptable in dev mode)" -ForegroundColor Gray
Write-Host "  - Run: .\fix-redis-connection.ps1 to diagnose" -ForegroundColor Yellow
Write-Host ""
Write-Host "Test completed!" -ForegroundColor Green
Write-Host ""
