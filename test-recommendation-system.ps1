# Test Recommendation System Flow
# Tests the complete flow from client request to personalized feed

param(
    [string]$UserId = "31ba8a23-8a4e-4b24-99c2-0d768e617e71",
    [string]$PostServiceUrl = "http://localhost:8085",
    [string]$RecommendServiceUrl = "http://localhost:8095",
    [string]$PythonServiceUrl = "http://localhost:8000"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RECOMMENDATION SYSTEM TEST" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check Python service health
Write-Host "[1] Testing Python Model Service..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "$PythonServiceUrl/health" -Method Get
    Write-Host "✅ Python service is healthy" -ForegroundColor Green
    Write-Host "   Version: $($healthResponse.model_version)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Python service is not available: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 2: Direct call to recommendation service
Write-Host "[2] Testing recommendation service directly..." -ForegroundColor Yellow
try {
    $recUrl = "$RecommendServiceUrl/api/recommendations/feed?userId=$UserId&page=0&size=10"
    Write-Host "   GET $recUrl" -ForegroundColor Gray
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $recResponse = Invoke-RestMethod -Uri $recUrl -Method Get
    $stopwatch.Stop()
    
    Write-Host "✅ Recommendations retrieved" -ForegroundColor Green
    Write-Host "   Count: $($recResponse.totalCount)" -ForegroundColor Gray
    Write-Host "   Time: $($stopwatch.ElapsedMilliseconds)ms" -ForegroundColor Gray
    Write-Host "   Source: $($recResponse.abVariant)" -ForegroundColor Gray
    
    if ($recResponse.recommendations) {
        Write-Host "   Top 5 recommendations with scores:" -ForegroundColor Gray
        $recResponse.recommendations | Select-Object -First 5 | ForEach-Object {
            Write-Host "      - $($_.postId): score=$([math]::Round($_.score, 4))" -ForegroundColor DarkGray
        }
        
        # Check score distribution
        $scores = $recResponse.recommendations | ForEach-Object { $_.score }
        $uniqueScores = $scores | Sort-Object -Unique
        
        if ($uniqueScores.Count -eq 1) {
            Write-Host "   ⚠️  WARNING: All posts have the same score ($($uniqueScores[0]))" -ForegroundColor Yellow
        } else {
            $avgScore = ($scores | Measure-Object -Average).Average
            $maxScore = ($scores | Measure-Object -Maximum).Maximum
            $minScore = ($scores | Measure-Object -Minimum).Minimum
            Write-Host "   Score range: $([math]::Round($minScore, 4)) - $([math]::Round($maxScore, 4)), avg=$([math]::Round($avgScore, 4))" -ForegroundColor DarkGray
        }
    }
} catch {
    Write-Host "❌ Failed to get recommendations: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: Test Python prediction endpoint with minimal data
Write-Host "[3] Testing Python prediction endpoint..." -ForegroundColor Yellow
try {
    $testRequest = @{
        userAcademic = @{
            userId = $UserId
            major = "Công nghệ thông tin"
            faculty = "Công nghệ thông tin và truyền thông"
            degree = "Đại học"
            batch = "K47"
        }
        userHistory = @()
        candidatePosts = @(
            @{
                postId = "test-post-1"
                content = "Thông báo về kỳ thi học kỳ 1 năm học 2024-2025"
                hashtags = @("hoctap", "thongbao")
                authorId = "author-1"
                authorMajor = "Công nghệ thông tin"
                authorFaculty = "Công nghệ thông tin và truyền thông"
                likeCount = 10
                commentCount = 5
                shareCount = 2
                viewCount = 100
            }
        )
        topK = 10
    }
    
    $jsonRequest = $testRequest | ConvertTo-Json -Depth 10
    Write-Host "   Request size: $($jsonRequest.Length) bytes" -ForegroundColor Gray
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $predResponse = Invoke-RestMethod -Uri "$PythonServiceUrl/api/model/predict" -Method Post -Body $jsonRequest -ContentType "application/json"
    $stopwatch.Stop()
    
    Write-Host "✅ Python prediction successful" -ForegroundColor Green
    Write-Host "   Ranked posts: $($predResponse.rankedPosts.Count)" -ForegroundColor Gray
    Write-Host "   Processing time: $($predResponse.processingTimeMs)ms" -ForegroundColor Gray
    Write-Host "   Model version: $($predResponse.modelVersion)" -ForegroundColor Gray
    
    if ($predResponse.rankedPosts) {
        Write-Host "   Ranked results:" -ForegroundColor Gray
        $predResponse.rankedPosts | ForEach-Object {
            Write-Host "      - $($_.postId): score=$([math]::Round($_.score, 4))" -ForegroundColor DarkGray
        }
    }
} catch {
    Write-Host "❌ Python prediction failed: $_" -ForegroundColor Red
    if ($_.ErrorDetails) {
        Write-Host "   Details: $($_.ErrorDetails.Message)" -ForegroundColor Red
    }
    if ($_.Exception.Response) {
        try {
            $reader = [System.IO.StreamReader]::new($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            Write-Host "   Response: $responseBody" -ForegroundColor Red
        } catch {}
    }
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Check logs for detailed information:" -ForegroundColor White
Write-Host "  - recommend-service: Check recommend-service/java-api/logs/" -ForegroundColor Gray
Write-Host "  - python-model: Check recommend-service/python-model/logs/" -ForegroundColor Gray
Write-Host ""
Write-Host "Database queries to run manually:" -ForegroundColor White
Write-Host "  # Check user feedback count" -ForegroundColor Gray
Write-Host "  docker exec -it postgres psql -U postgres -d recommend_db -c 'SELECT COUNT(*) FROM user_feedback;'" -ForegroundColor DarkGray
Write-Host "  # Check post embeddings" -ForegroundColor Gray
Write-Host "  docker exec -it postgres psql -U postgres -d recommend_db -c 'SELECT post_id, like_count, score FROM post_embeddings LIMIT 5;'" -ForegroundColor DarkGray
Write-Host "  # Check recent interactions" -ForegroundColor Gray
Write-Host "  docker exec -it postgres psql -U postgres -d recommend_db -c 'SELECT user_id, post_id, feedback_type, timestamp FROM user_feedback ORDER BY timestamp DESC LIMIT 5;'" -ForegroundColor DarkGray
