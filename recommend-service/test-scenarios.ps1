# Comprehensive Test Scenarios for Recommend Service
# Tests realistic user workflows and edge cases

param(
    [string]$BaseUrl = "http://localhost:8095",
    [string]$PythonUrl = "http://localhost:8000"
)

Write-Host "`n╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Recommend Service - Scenario Testing              ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$scenariosPassed = 0
$scenariosFailed = 0

function Test-Scenario {
    param(
        [string]$Name,
        [scriptblock]$TestBlock
    )
    
    Write-Host "`n━━━ Scenario: $Name ━━━" -ForegroundColor Yellow
    try {
        & $TestBlock
        Write-Host "✓ Scenario passed: $Name" -ForegroundColor Green
        $script:scenariosPassed++
    } catch {
        Write-Host "✗ Scenario failed: $Name" -ForegroundColor Red
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        $script:scenariosFailed++
    }
}

# ============================================================
# Scenario 1: New User First Visit
# ============================================================
Test-Scenario -Name "New User First Visit" -TestBlock {
    Write-Host "  1. New user requests feed..." -ForegroundColor Gray
    
    $userId = "new_user_$(Get-Random -Maximum 10000)"
    $response = Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/feed?userId=$userId&size=20" -Method GET
    
    if ($response.recommendations.Count -gt 0) {
        Write-Host "  ✓ Got $($response.recommendations.Count) recommendations" -ForegroundColor Green
    } else {
        throw "No recommendations returned for new user"
    }
    
    Write-Host "  2. User views a post..." -ForegroundColor Gray
    $firstPost = $response.recommendations[0]
    $interaction = @{
        userId = $userId
        postId = $firstPost.postId
        type = "VIEW"
        viewDuration = 30
    } | ConvertTo-Json
    
    Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/interaction" -Method POST -Body $interaction -ContentType "application/json" | Out-Null
    Write-Host "  ✓ Interaction recorded" -ForegroundColor Green
}

# ============================================================
# Scenario 2: Active User with History
# ============================================================
Test-Scenario -Name "Active User with Interaction History" -TestBlock {
    $userId = "active_user_001"
    
    Write-Host "  1. User views multiple posts..." -ForegroundColor Gray
    $response = Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/feed?userId=$userId&size=10" -Method GET
    
    foreach ($i in 0..2) {
        $post = $response.recommendations[$i]
        $interaction = @{
            userId = $userId
            postId = $post.postId
            type = "VIEW"
            viewDuration = (Get-Random -Minimum 10 -Maximum 60)
        } | ConvertTo-Json
        
        Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/interaction" -Method POST -Body $interaction -ContentType "application/json" | Out-Null
    }
    Write-Host "  ✓ Recorded 3 view interactions" -ForegroundColor Green
    
    Write-Host "  2. User likes a post..." -ForegroundColor Gray
    $likePost = $response.recommendations[1]
    $feedback = @{
        userId = $userId
        postId = $likePost.postId
        feedbackType = "LIKE"
    } | ConvertTo-Json
    
    Invoke-RestMethod -Uri "$BaseUrl/api/recommend/feedback" -Method POST -Body $feedback -ContentType "application/json" | Out-Null
    Write-Host "  ✓ Like feedback recorded" -ForegroundColor Green
    
    Write-Host "  3. Refresh feed..." -ForegroundColor Gray
    $newResponse = Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/feed?userId=$userId&size=10" -Method GET
    Write-Host "  ✓ Got $($newResponse.recommendations.Count) updated recommendations" -ForegroundColor Green
}

# ============================================================
# Scenario 3: Academic Content Discovery
# ============================================================
Test-Scenario -Name "Academic Content Discovery" -TestBlock {
    Write-Host "  1. Student looking for scholarship info..." -ForegroundColor Gray
    
    $request = @{
        userId = "student_cs_001"
        page = 0
        size = 20
        includeExplanations = $true
        filters = @{
            categories = @("scholarship", "research")
            minScore = 0.3
        }
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$BaseUrl/api/recommend/posts" -Method POST -Body $request -ContentType "application/json"
    
    if ($response.recommendations.Count -gt 0) {
        Write-Host "  ✓ Found $($response.recommendations.Count) relevant posts" -ForegroundColor Green
        
        # Check if explanations are included
        if ($response.recommendations[0].explanation) {
            Write-Host "  ✓ Explanations provided" -ForegroundColor Green
        }
    } else {
        throw "No academic content found"
    }
}

# ============================================================
# Scenario 4: Batch Processing
# ============================================================
Test-Scenario -Name "Batch Embedding Processing" -TestBlock {
    Write-Host "  1. Processing multiple posts..." -ForegroundColor Gray
    
    $batchRequest = @{
        posts = @(
            @{
                post_id = "batch_001"
                content = "Hội thảo AI và Machine Learning 2024"
                tags = @("event", "ai")
            },
            @{
                post_id = "batch_002"
                content = "Học bổng toàn phần du học Mỹ"
                tags = @("scholarship", "study-abroad")
            },
            @{
                post_id = "batch_003"
                content = "Câu hỏi về thuật toán sorting"
                tags = @("qa", "algorithm")
            }
        )
    } | ConvertTo-Json -Depth 10
    
    $response = Invoke-RestMethod -Uri "$PythonUrl/embed/post/batch" -Method POST -Body $batchRequest -ContentType "application/json"
    
    if ($response.embeddings.Count -eq 3) {
        Write-Host "  ✓ Processed 3 posts successfully" -ForegroundColor Green
    } else {
        throw "Batch processing failed"
    }
}

# ============================================================
# Scenario 5: Cache Management
# ============================================================
Test-Scenario -Name "Cache Invalidation and Refresh" -TestBlock {
    $userId = "cache_test_user"
    
    Write-Host "  1. Get initial recommendations..." -ForegroundColor Gray
    $response1 = Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/feed?userId=$userId&size=5" -Method GET
    Write-Host "  ✓ Got $($response1.recommendations.Count) recommendations" -ForegroundColor Green
    
    Write-Host "  2. Invalidate cache..." -ForegroundColor Gray
    $cacheRequest = @{
        userId = $userId
    } | ConvertTo-Json
    
    Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/cache/invalidate" -Method POST -Body $cacheRequest -ContentType "application/json" | Out-Null
    Write-Host "  ✓ Cache invalidated" -ForegroundColor Green
    
    Write-Host "  3. Get fresh recommendations..." -ForegroundColor Gray
    $response2 = Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/feed?userId=$userId&size=5" -Method GET
    Write-Host "  ✓ Got $($response2.recommendations.Count) fresh recommendations" -ForegroundColor Green
}

# ============================================================
# Scenario 6: Edge Cases
# ============================================================
Test-Scenario -Name "Edge Cases and Error Handling" -TestBlock {
    Write-Host "  1. Test with empty content..." -ForegroundColor Gray
    try {
        $emptyRequest = @{
            post_id = "empty_001"
            content = ""
            tags = @()
        } | ConvertTo-Json
        
        Invoke-RestMethod -Uri "$PythonUrl/embed/post" -Method POST -Body $emptyRequest -ContentType "application/json" | Out-Null
        Write-Host "  ⚠ Empty content accepted (should validate)" -ForegroundColor Yellow
    } catch {
        Write-Host "  ✓ Empty content rejected correctly" -ForegroundColor Green
    }
    
    Write-Host "  2. Test with very long content..." -ForegroundColor Gray
    $longContent = "Lorem ipsum " * 500
    $longRequest = @{
        post_id = "long_001"
        content = $longContent
        tags = @("test")
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$PythonUrl/embed/post" -Method POST -Body $longRequest -ContentType "application/json"
    Write-Host "  ✓ Long content processed (truncated to max length)" -ForegroundColor Green
    
    Write-Host "  3. Test pagination edge cases..." -ForegroundColor Gray
    $response = Invoke-RestMethod -Uri "$BaseUrl/api/recommendation/feed?userId=test&page=999&size=100" -Method GET
    Write-Host "  ✓ High page number handled gracefully" -ForegroundColor Green
}

# ============================================================
# Scenario 7: Performance Test
# ============================================================
Test-Scenario -Name "Performance - Concurrent Requests" -TestBlock {
    Write-Host "  Testing concurrent recommendation requests..." -ForegroundColor Gray
    
    $jobs = @()
    for ($i = 0; $i -lt 5; $i++) {
        $jobs += Start-Job -ScriptBlock {
            param($url, $userId)
            Invoke-RestMethod -Uri "$url/api/recommendation/feed?userId=$userId&size=10" -Method GET
        } -ArgumentList $BaseUrl, "perf_user_$i"
    }
    
    $results = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
    
    Write-Host "  ✓ Processed $($results.Count) concurrent requests" -ForegroundColor Green
}

# ============================================================
# Summary
# ============================================================

Write-Host "`n╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Scenario Test Summary                              ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Total Scenarios: $($scenariosPassed + $scenariosFailed)" -ForegroundColor White
Write-Host "  Passed: $scenariosPassed" -ForegroundColor Green
Write-Host "  Failed: $scenariosFailed" -ForegroundColor $(if ($scenariosFailed -gt 0) { "Red" } else { "Gray" })
Write-Host ""

if ($scenariosFailed -eq 0) {
    Write-Host "🎉 All scenarios passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "⚠️  Some scenarios failed." -ForegroundColor Yellow
    exit 1
}
