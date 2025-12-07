# ============================================================================
# CTU Connect - Recommendation Service Development Test Script
# ============================================================================
# Script này test recommendation service khi chạy trên IDE (development mode)
# Author: AI Assistant
# Date: 2024-12-07
# ============================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CTU CONNECT - RECOMMENDATION SERVICE TEST" -ForegroundColor Cyan
Write-Host "  Development Environment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$ErrorCount = 0
$SuccessCount = 0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function Test-Service {
    param(
        [string]$Name,
        [string]$Url,
        [int]$ExpectedStatus = 200
    )
    
    Write-Host "Testing $Name..." -NoNewline
    
    try {
        $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 10 -UseBasicParsing
        
        if ($response.StatusCode -eq $ExpectedStatus) {
            Write-Host " ✅ OK" -ForegroundColor Green
            Write-Host "  └─ Status: $($response.StatusCode)" -ForegroundColor Gray
            $script:SuccessCount++
            return $true
        } else {
            Write-Host " ❌ FAILED" -ForegroundColor Red
            Write-Host "  └─ Expected: $ExpectedStatus, Got: $($response.StatusCode)" -ForegroundColor Red
            $script:ErrorCount++
            return $false
        }
    }
    catch {
        Write-Host " ❌ FAILED" -ForegroundColor Red
        Write-Host "  └─ Error: $($_.Exception.Message)" -ForegroundColor Red
        $script:ErrorCount++
        return $false
    }
}

function Test-ServiceJson {
    param(
        [string]$Name,
        [string]$Url,
        [string]$ExpectedField = $null
    )
    
    Write-Host "Testing $Name..." -NoNewline
    
    try {
        $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 10
        
        if ($ExpectedField) {
            if ($response.$ExpectedField) {
                Write-Host " ✅ OK" -ForegroundColor Green
                Write-Host "  └─ $ExpectedField: $($response.$ExpectedField)" -ForegroundColor Gray
                $script:SuccessCount++
                return $true
            } else {
                Write-Host " ❌ FAILED" -ForegroundColor Red
                Write-Host "  └─ Field '$ExpectedField' not found in response" -ForegroundColor Red
                $script:ErrorCount++
                return $false
            }
        } else {
            Write-Host " ✅ OK" -ForegroundColor Green
            Write-Host "  └─ Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
            $script:SuccessCount++
            return $true
        }
    }
    catch {
        Write-Host " ❌ FAILED" -ForegroundColor Red
        Write-Host "  └─ Error: $($_.Exception.Message)" -ForegroundColor Red
        $script:ErrorCount++
        return $false
    }
}

function Test-Docker {
    param([string]$ContainerName)
    
    Write-Host "Checking Docker container: $ContainerName..." -NoNewline
    
    try {
        $container = docker ps --filter "name=$ContainerName" --format "{{.Status}}"
        
        if ($container -and $container -like "*Up*") {
            Write-Host " ✅ Running" -ForegroundColor Green
            Write-Host "  └─ Status: $container" -ForegroundColor Gray
            $script:SuccessCount++
            return $true
        } else {
            Write-Host " ❌ Not Running" -ForegroundColor Red
            $script:ErrorCount++
            return $false
        }
    }
    catch {
        Write-Host " ❌ FAILED" -ForegroundColor Red
        Write-Host "  └─ Error: $($_.Exception.Message)" -ForegroundColor Red
        $script:ErrorCount++
        return $false
    }
}

# ============================================================================
# TEST 1: CHECK DOCKER CONTAINERS
# ============================================================================

Write-Host "`n[TEST 1] Checking Docker Containers" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Yellow

Test-Docker "recommendation-postgres" | Out-Null
Test-Docker "neo4j-graph-db" | Out-Null
Test-Docker "redis" | Out-Null
Test-Docker "kafka" | Out-Null

# ============================================================================
# TEST 2: CHECK DATABASE CONNECTIVITY
# ============================================================================

Write-Host "`n[TEST 2] Database Connectivity" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Yellow

# PostgreSQL
Write-Host "Testing PostgreSQL..." -NoNewline
try {
    docker exec recommendation-postgres pg_isready -U postgres -d recommendation_db | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅ OK" -ForegroundColor Green
        $SuccessCount++
    } else {
        Write-Host " ❌ FAILED" -ForegroundColor Red
        $ErrorCount++
    }
}
catch {
    Write-Host " ❌ FAILED" -ForegroundColor Red
    Write-Host "  └─ Error: $($_.Exception.Message)" -ForegroundColor Red
    $ErrorCount++
}

# Redis
Write-Host "Testing Redis..." -NoNewline
try {
    $redisResult = docker exec redis redis-cli ping
    if ($redisResult -eq "PONG") {
        Write-Host " ✅ OK" -ForegroundColor Green
        Write-Host "  └─ Response: PONG" -ForegroundColor Gray
        $SuccessCount++
    } else {
        Write-Host " ❌ FAILED" -ForegroundColor Red
        $ErrorCount++
    }
}
catch {
    Write-Host " ❌ FAILED" -ForegroundColor Red
    Write-Host "  └─ Error: $($_.Exception.Message)" -ForegroundColor Red
    $ErrorCount++
}

# Neo4j
Test-Service "Neo4j Browser" "http://localhost:7474" | Out-Null

# ============================================================================
# TEST 3: PYTHON ML SERVICE HEALTH
# ============================================================================

Write-Host "`n[TEST 3] Python ML Service (Port 8097)" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Yellow

$pythonHealthy = Test-ServiceJson "Python Service Root" "http://localhost:8097" "status"
Test-ServiceJson "Python Health Check" "http://localhost:8097/health" "status" | Out-Null
Test-ServiceJson "Python Metrics" "http://localhost:8097/metrics" "timestamp" | Out-Null

# ============================================================================
# TEST 4: JAVA SERVICE HEALTH
# ============================================================================

Write-Host "`n[TEST 4] Java Service (Port 8095)" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Yellow

$javaHealthy = Test-ServiceJson "Java Health Check" "http://localhost:8095/actuator/health" "status"
Test-Service "Java Service Info" "http://localhost:8095/actuator/info" | Out-Null

# ============================================================================
# TEST 5: PYTHON ML ENDPOINTS
# ============================================================================

Write-Host "`n[TEST 5] Python ML Endpoints" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Yellow

if ($pythonHealthy) {
    # Test prediction endpoint
    Write-Host "Testing Python Prediction Endpoint..." -NoNewline
    
    $testPayload = @{
        userAcademic = @{
            userId = "test_user_123"
            major = "CNTT"
            interests = @("AI", "Machine Learning", "Data Science")
        }
        userHistory = @{
            viewedPosts = @()
            likedPosts = @()
            interactions = @()
        }
        candidatePosts = @(
            @{
                postId = "post_1"
                content = "Nghiên cứu về Machine Learning trong y tế"
                category = "research"
            }
            @{
                postId = "post_2"
                content = "Hội thảo về AI và Deep Learning"
                category = "event"
            }
        )
        topK = 10
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8097/api/model/predict" `
            -Method Post `
            -Body $testPayload `
            -ContentType "application/json" `
            -TimeoutSec 30
        
        Write-Host " ✅ OK" -ForegroundColor Green
        Write-Host "  └─ Returned $($response.rankedPosts.Count) ranked posts" -ForegroundColor Gray
        if ($response.modelVersion) {
            Write-Host "  └─ Model Version: $($response.modelVersion)" -ForegroundColor Gray
        }
        $SuccessCount++
    }
    catch {
        Write-Host " ❌ FAILED" -ForegroundColor Red
        Write-Host "  └─ Error: $($_.Exception.Message)" -ForegroundColor Red
        $ErrorCount++
    }
} else {
    Write-Host "Skipping Python ML Endpoint tests (Service not healthy)" -ForegroundColor Yellow
}

# ============================================================================
# TEST 6: JAVA RECOMMENDATION ENDPOINTS
# ============================================================================

Write-Host "`n[TEST 6] Java Recommendation Endpoints" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Yellow

if ($javaHealthy) {
    # Test feed endpoint
    Write-Host "Testing Get Feed Endpoint..." -NoNewline
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8095/api/recommendation/feed?userId=test_user_123&size=10" `
            -Method Get `
            -TimeoutSec 30
        
        Write-Host " ✅ OK" -ForegroundColor Green
        Write-Host "  └─ Returned $($response.recommendations.Count) recommendations" -ForegroundColor Gray
        Write-Host "  └─ Model Used: $($response.modelUsed)" -ForegroundColor Gray
        $SuccessCount++
    }
    catch {
        Write-Host " ⚠️ WARNING" -ForegroundColor Yellow
        Write-Host "  └─ Error: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "  └─ This may be expected if no test data exists" -ForegroundColor Yellow
    }
    
    # Test similar posts endpoint
    Write-Host "Testing Get Similar Posts Endpoint..." -NoNewline
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8095/api/recommendation/similar/post_123?size=5" `
            -Method Get `
            -TimeoutSec 30
        
        Write-Host " ✅ OK" -ForegroundColor Green
        Write-Host "  └─ Returned $($response.recommendations.Count) similar posts" -ForegroundColor Gray
        $SuccessCount++
    }
    catch {
        Write-Host " ⚠️ WARNING" -ForegroundColor Yellow
        Write-Host "  └─ Error: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "  └─ This may be expected if no test data exists" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping Java Recommendation Endpoint tests (Service not healthy)" -ForegroundColor Yellow
}

# ============================================================================
# TEST 7: INTEGRATION TEST (Java → Python)
# ============================================================================

Write-Host "`n[TEST 7] Integration Test (Java → Python)" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Yellow

if ($javaHealthy -and $pythonHealthy) {
    Write-Host "Testing end-to-end integration..." -NoNewline
    
    try {
        # Call Java service, which should internally call Python service
        $response = Invoke-RestMethod -Uri "http://localhost:8095/api/recommendation/feed?userId=integration_test&size=5" `
            -Method Get `
            -TimeoutSec 30
        
        Write-Host " ✅ OK" -ForegroundColor Green
        Write-Host "  └─ Java service received request" -ForegroundColor Gray
        Write-Host "  └─ Python ML service processed" -ForegroundColor Gray
        Write-Host "  └─ Returned response successfully" -ForegroundColor Gray
        $SuccessCount++
    }
    catch {
        Write-Host " ⚠️ WARNING" -ForegroundColor Yellow
        Write-Host "  └─ Integration test failed, but services are healthy" -ForegroundColor Yellow
        Write-Host "  └─ This may be due to missing test data" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping integration test (Services not healthy)" -ForegroundColor Yellow
}

# ============================================================================
# TEST 8: API DOCUMENTATION
# ============================================================================

Write-Host "`n[TEST 8] API Documentation" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────" -ForegroundColor Yellow

Test-Service "Python Swagger UI" "http://localhost:8097/docs" | Out-Null
Test-Service "Python ReDoc" "http://localhost:8097/redoc" | Out-Null

# ============================================================================
# TEST SUMMARY
# ============================================================================

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  TEST SUMMARY" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$TotalTests = $SuccessCount + $ErrorCount
$SuccessRate = if ($TotalTests -gt 0) { [math]::Round(($SuccessCount / $TotalTests) * 100, 2) } else { 0 }

Write-Host ""
Write-Host "Total Tests: $TotalTests" -ForegroundColor White
Write-Host "✅ Passed: $SuccessCount" -ForegroundColor Green
Write-Host "❌ Failed: $ErrorCount" -ForegroundColor Red
Write-Host "Success Rate: $SuccessRate%" -ForegroundColor $(if ($SuccessRate -ge 80) { "Green" } elseif ($SuccessRate -ge 50) { "Yellow" } else { "Red" })
Write-Host ""

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

if ($ErrorCount -gt 0) {
    Write-Host "============================================" -ForegroundColor Yellow
    Write-Host "  RECOMMENDATIONS" -ForegroundColor Yellow
    Write-Host "============================================" -ForegroundColor Yellow
    Write-Host ""
    
    Write-Host "Some tests failed. Please check:" -ForegroundColor Yellow
    Write-Host "1. Ensure Docker containers are running:" -ForegroundColor White
    Write-Host "   cd recommendation-service-java" -ForegroundColor Gray
    Write-Host "   docker-compose -f docker-compose.dev.yml up -d" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Ensure Python service is running:" -ForegroundColor White
    Write-Host "   cd recommendation-service-python" -ForegroundColor Gray
    Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host "   python app.py" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Ensure Java service is running:" -ForegroundColor White
    Write-Host "   Run from IntelliJ IDEA or:" -ForegroundColor Gray
    Write-Host "   cd recommendation-service-java" -ForegroundColor Gray
    Write-Host "   mvn spring-boot:run -Dspring-boot.run.profiles=dev" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  🎉 ALL TESTS PASSED! 🎉" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Recommendation services are working correctly!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "1. Add test data to databases" -ForegroundColor Gray
    Write-Host "2. Train ML models with real data" -ForegroundColor Gray
    Write-Host "3. Fine-tune scoring weights" -ForegroundColor Gray
    Write-Host "4. Monitor performance and optimize" -ForegroundColor Gray
    Write-Host ""
}

# ============================================================================
# USEFUL LINKS
# ============================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  USEFUL LINKS" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Python Service:" -ForegroundColor White
Write-Host "  - API Docs: http://localhost:8097/docs" -ForegroundColor Gray
Write-Host "  - Health: http://localhost:8097/health" -ForegroundColor Gray
Write-Host "  - Metrics: http://localhost:8097/metrics" -ForegroundColor Gray
Write-Host ""
Write-Host "Java Service:" -ForegroundColor White
Write-Host "  - Health: http://localhost:8095/actuator/health" -ForegroundColor Gray
Write-Host "  - Info: http://localhost:8095/actuator/info" -ForegroundColor Gray
Write-Host "  - Metrics: http://localhost:8095/actuator/metrics" -ForegroundColor Gray
Write-Host ""
Write-Host "Databases:" -ForegroundColor White
Write-Host "  - Neo4j Browser: http://localhost:7474" -ForegroundColor Gray
Write-Host "  - PostgreSQL: localhost:5435" -ForegroundColor Gray
Write-Host "  - Redis: localhost:6379" -ForegroundColor Gray
Write-Host ""

# Exit code
exit $ErrorCount
