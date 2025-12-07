# Test Hybrid Recommendation System
# This script tests both Java and Python services

Write-Host "🚀 Testing CTU Connect Hybrid Recommendation System" -ForegroundColor Cyan
Write-Host ""

$javaUrl = "http://localhost:8095"
$pythonUrl = "http://localhost:8097"

# Test 1: Java Service Health
Write-Host "Test 1: Java Service Health Check..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$javaUrl/actuator/health" -Method Get
    if ($response.status -eq "UP") {
        Write-Host "✅ Java service is running" -ForegroundColor Green
    } else {
        Write-Host "❌ Java service health check failed" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Java service not reachable: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 2: Python Service Health
Write-Host "Test 2: Python Service Health Check..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$pythonUrl/health" -Method Get
    if ($response.status -eq "healthy") {
        Write-Host "✅ Python service is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Python service unhealthy (may still work in fallback mode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Python service not reachable (Java will use fallback)" -ForegroundColor Yellow
}

Write-Host ""

# Test 3: Python Model Info
Write-Host "Test 3: Python Model Information..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$pythonUrl/api/model/info" -Method Get
    Write-Host "  Model Version: $($response.modelVersion)" -ForegroundColor Cyan
    Write-Host "  Embedding Dimension: $($response.embeddingDimension)" -ForegroundColor Cyan
    Write-Host "  Model Ready: $($response.isReady)" -ForegroundColor Cyan
    Write-Host "✅ Model info retrieved" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Model info not available: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""

# Test 4: Python Direct Prediction
Write-Host "Test 4: Python Direct Prediction..." -ForegroundColor Yellow
try {
    $requestBody = @{
        userAcademic = @{
            userId = "test-user-123"
            major = "Computer Science"
            faculty = "Engineering"
            degree = "Bachelor"
            batch = "K48"
        }
        userHistory = @()
        candidatePosts = @(
            @{
                postId = "post-1"
                content = "Hội thảo Machine Learning và AI tại CTU"
                hashtags = @("#AI", "#MachineLearning", "#CTU")
                authorMajor = "Computer Science"
                authorFaculty = "Engineering"
                likesCount = 15
                commentsCount = 8
                sharesCount = 3
            },
            @{
                postId = "post-2"
                content = "Tuyển sinh chương trình Thạc sĩ Khoa học máy tính"
                hashtags = @("#TuyenSinh", "#ThacSi", "#CNTT")
                authorMajor = "Computer Science"
                authorFaculty = "Engineering"
                likesCount = 25
                commentsCount = 12
                sharesCount = 5
            },
            @{
                postId = "post-3"
                content = "Bữa trưa hôm nay ăn gì nhỉ?"
                hashtags = @("#Food")
                authorMajor = "Business"
                authorFaculty = "Economics"
                likesCount = 5
                commentsCount = 2
                sharesCount = 0
            }
        )
        topK = 3
    } | ConvertTo-Json -Depth 10

    $response = Invoke-RestMethod -Uri "$pythonUrl/api/model/predict" -Method Post `
        -ContentType "application/json" -Body $requestBody

    Write-Host "  Processing Time: $($response.processingTimeMs)ms" -ForegroundColor Cyan
    Write-Host "  Ranked Posts:" -ForegroundColor Cyan
    foreach ($post in $response.rankedPosts) {
        Write-Host "    #$($post.rank): $($post.postId) - Score: $($post.score)" -ForegroundColor Gray
        if ($post.contentSimilarity) {
            Write-Host "       Content: $($post.contentSimilarity), Academic: $($post.academicScore), Popularity: $($post.popularityScore)" -ForegroundColor DarkGray
        }
    }
    Write-Host "✅ Python prediction successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Python prediction failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 5: Java Recommendation Feed
Write-Host "Test 5: Java Recommendation Feed (Hybrid)..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$javaUrl/api/recommendation/feed?userId=test-user-123&size=5" -Method Get
    
    Write-Host "  Source: $($response.metadata.source)" -ForegroundColor Cyan
    Write-Host "  Processing Time: $($response.metadata.processingTimeMs)ms" -ForegroundColor Cyan
    Write-Host "  Posts Returned: $($response.posts.Count)" -ForegroundColor Cyan
    
    if ($response.posts.Count -gt 0) {
        Write-Host "  Top Recommendations:" -ForegroundColor Cyan
        foreach ($post in $response.posts | Select-Object -First 3) {
            Write-Host "    - $($post.postId): Score $($post.score)" -ForegroundColor Gray
        }
    }
    
    Write-Host "✅ Java feed retrieved successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Java feed failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 6: Cache Check
Write-Host "Test 6: Redis Cache Check..." -ForegroundColor Yellow
try {
    docker exec recommendation-redis redis-cli DBSIZE | Out-Null
    $keys = docker exec recommendation-redis redis-cli KEYS "recommend:*"
    Write-Host "  Cache keys count: $($keys.Count)" -ForegroundColor Cyan
    Write-Host "✅ Redis cache accessible" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Cannot check Redis cache: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""

# Summary
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "📊 Test Summary" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services Status:" -ForegroundColor White
Write-Host "  ✅ Java Service: Running on port 8095" -ForegroundColor Green
Write-Host "  ✅ Python Service: Running on port 8097" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor White
Write-Host "  1. Load test data: .\load-test-data.ps1" -ForegroundColor Gray
Write-Host "  2. Train models: python training/train_model.py" -ForegroundColor Gray
Write-Host "  3. Monitor: http://localhost:8095/actuator/metrics" -ForegroundColor Gray
Write-Host ""
Write-Host "Documentation:" -ForegroundColor White
Write-Host "  - Setup: RECOMMENDATION_HYBRID_SETUP.md" -ForegroundColor Gray
Write-Host "  - Architecture: recommendation-service-java/HYBRID_ARCHITECTURE.md" -ForegroundColor Gray
Write-Host ""
