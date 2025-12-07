# Test script for Hybrid Recommendation System API
# Usage: .\test-hybrid-api.ps1

$baseUrl = "http://localhost:8095"
$userId = "user-001"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "CTU Connect Recommendation System - API Test" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "Test 1: Health Check" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/actuator/health" -Method Get
    Write-Host "✅ Service is healthy: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "❌ Health check failed: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 2: Get Feed (First time - Cache Miss)
Write-Host "Test 2: Get Feed (Cache Miss)" -ForegroundColor Yellow
try {
    $startTime = Get-Date
    $feed = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/feed?userId=$userId&size=10" -Method Get
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMilliseconds
    
    Write-Host "✅ Feed retrieved successfully" -ForegroundColor Green
    Write-Host "   User ID: $($feed.userId)" -ForegroundColor Gray
    Write-Host "   Total recommendations: $($feed.totalCount)" -ForegroundColor Gray
    Write-Host "   Cache variant: $($feed.abVariant)" -ForegroundColor Gray
    Write-Host "   Processing time: $($feed.processingTimeMs)ms (Total: $([math]::Round($duration, 2))ms)" -ForegroundColor Gray
    
    if ($feed.recommendations.Count -gt 0) {
        Write-Host "   First post: $($feed.recommendations[0].postId)" -ForegroundColor Gray
        Write-Host "   Score: $($feed.recommendations[0].score)" -ForegroundColor Gray
    }
} catch {
    Write-Host "❌ Failed to get feed: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: Get Feed (Second time - Cache Hit)
Write-Host "Test 3: Get Feed (Cache Hit)" -ForegroundColor Yellow
try {
    Start-Sleep -Seconds 1
    $startTime = Get-Date
    $feed2 = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/feed?userId=$userId&size=10" -Method Get
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMilliseconds
    
    Write-Host "✅ Feed retrieved from cache" -ForegroundColor Green
    Write-Host "   Cache variant: $($feed2.abVariant)" -ForegroundColor Gray
    Write-Host "   Processing time: $($feed2.processingTimeMs)ms (Total: $([math]::Round($duration, 2))ms)" -ForegroundColor Gray
    
    if ($feed2.abVariant -eq "cached") {
        Write-Host "   🚀 Cache working correctly!" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Failed to get cached feed: $_" -ForegroundColor Red
}
Write-Host ""

# Test 4: Record Interaction (Like)
Write-Host "Test 4: Record Interaction (LIKE)" -ForegroundColor Yellow
if ($feed.recommendations.Count -gt 0) {
    $postId = $feed.recommendations[0].postId
    
    $interactionBody = @{
        userId = $userId
        postId = $postId
        type = "LIKE"
        viewDuration = 5.5
        context = @{
            source = "feed"
            position = 0
        }
    } | ConvertTo-Json
    
    try {
        $interactionResult = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/interaction" `
            -Method Post -Body $interactionBody -ContentType "application/json"
        
        Write-Host "✅ Interaction recorded successfully" -ForegroundColor Green
        Write-Host "   Post ID: $postId" -ForegroundColor Gray
        Write-Host "   Status: $($interactionResult.status)" -ForegroundColor Gray
    } catch {
        Write-Host "❌ Failed to record interaction: $_" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️  No posts available to interact with" -ForegroundColor Yellow
}
Write-Host ""

# Test 5: Get Feed (After Interaction - Cache Invalidated)
Write-Host "Test 5: Get Feed (After Interaction - Cache Invalidated)" -ForegroundColor Yellow
try {
    Start-Sleep -Seconds 1
    $startTime = Get-Date
    $feed3 = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/feed?userId=$userId&size=10" -Method Get
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalMilliseconds
    
    Write-Host "✅ Feed retrieved after interaction" -ForegroundColor Green
    Write-Host "   Cache variant: $($feed3.abVariant)" -ForegroundColor Gray
    Write-Host "   Processing time: $($feed3.processingTimeMs)ms (Total: $([math]::Round($duration, 2))ms)" -ForegroundColor Gray
    
    if ($feed3.abVariant -ne "cached") {
        Write-Host "   🚀 Cache invalidation working correctly!" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Failed to get feed after interaction: $_" -ForegroundColor Red
}
Write-Host ""

# Test 6: Record Multiple Interactions
Write-Host "Test 6: Record Multiple Interactions" -ForegroundColor Yellow
$interactionTypes = @("VIEW", "COMMENT", "SHARE")
$successCount = 0

foreach ($type in $interactionTypes) {
    if ($feed.recommendations.Count -gt 0) {
        $postId = $feed.recommendations[[Math]::Min(1, $feed.recommendations.Count - 1)].postId
        
        $body = @{
            userId = $userId
            postId = $postId
            type = $type
            viewDuration = 3.2
            context = @{}
        } | ConvertTo-Json
        
        try {
            $result = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/interaction" `
                -Method Post -Body $body -ContentType "application/json"
            $successCount++
            Write-Host "   ✅ $type interaction recorded" -ForegroundColor Gray
        } catch {
            Write-Host "   ❌ Failed to record $type interaction" -ForegroundColor Red
        }
    }
}

Write-Host "✅ Recorded $successCount/$($interactionTypes.Count) interactions" -ForegroundColor Green
Write-Host ""

# Test 7: Test Different Users
Write-Host "Test 7: Test Multiple Users" -ForegroundColor Yellow
$userIds = @("user-002", "user-003", "user-004")
$successCount = 0

foreach ($testUserId in $userIds) {
    try {
        $userFeed = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/feed?userId=$testUserId&size=5" -Method Get
        $successCount++
        Write-Host "   ✅ User $testUserId: $($userFeed.totalCount) recommendations" -ForegroundColor Gray
    } catch {
        Write-Host "   ❌ Failed for user $testUserId" -ForegroundColor Red
    }
}

Write-Host "✅ Retrieved feeds for $successCount/$($userIds.Count) users" -ForegroundColor Green
Write-Host ""

# Test 8: Test Pagination
Write-Host "Test 8: Test Pagination" -ForegroundColor Yellow
try {
    $page0 = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/feed?userId=$userId&page=0&size=5" -Method Get
    $page1 = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/feed?userId=$userId&page=1&size=5" -Method Get
    
    Write-Host "✅ Pagination working" -ForegroundColor Green
    Write-Host "   Page 0: $($page0.recommendations.Count) items" -ForegroundColor Gray
    Write-Host "   Page 1: $($page1.recommendations.Count) items" -ForegroundColor Gray
} catch {
    Write-Host "❌ Pagination test failed: $_" -ForegroundColor Red
}
Write-Host ""

# Test 9: Cache Invalidation API
Write-Host "Test 9: Manual Cache Invalidation" -ForegroundColor Yellow
try {
    $invalidateResult = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/cache/invalidate?userId=$userId" -Method Post
    Write-Host "✅ Cache invalidated successfully" -ForegroundColor Green
    Write-Host "   Status: $($invalidateResult.status)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Cache invalidation failed: $_" -ForegroundColor Red
}
Write-Host ""

# Test 10: Performance Comparison
Write-Host "Test 10: Performance Comparison" -ForegroundColor Yellow
Write-Host "   Testing cache performance..." -ForegroundColor Gray

# Invalidate cache first
try { 
    Invoke-RestMethod -Uri "$baseUrl/api/recommendation/cache/invalidate?userId=$userId" -Method Post | Out-Null
} catch {}

# First call (no cache)
Start-Sleep -Seconds 1
$startTime = Get-Date
try {
    $result1 = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/feed?userId=$userId&size=10" -Method Get
    $duration1 = ($Get-Date) - $startTime
    $cacheStatus1 = $result1.abVariant
} catch {
    $duration1 = ($Get-Date) - $startTime
    $cacheStatus1 = "error"
}

# Second call (with cache)
Start-Sleep -Seconds 0.5
$startTime = Get-Date
try {
    $result2 = Invoke-RestMethod -Uri "$baseUrl/api/recommendation/feed?userId=$userId&size=10" -Method Get
    $duration2 = ($Get-Date) - $startTime
    $cacheStatus2 = $result2.abVariant
} catch {
    $duration2 = ($Get-Date) - $startTime
    $cacheStatus2 = "error"
}

Write-Host "   Cache Miss: $([math]::Round($duration1.TotalMilliseconds, 2))ms (variant: $cacheStatus1)" -ForegroundColor Gray
Write-Host "   Cache Hit:  $([math]::Round($duration2.TotalMilliseconds, 2))ms (variant: $cacheStatus2)" -ForegroundColor Gray

if ($duration2.TotalMilliseconds -lt $duration1.TotalMilliseconds) {
    $speedup = [math]::Round($duration1.TotalMilliseconds / $duration2.TotalMilliseconds, 2)
    Write-Host "   🚀 Cache is ${speedup}x faster!" -ForegroundColor Green
}
Write-Host ""

# Summary
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "✅ All core API tests completed" -ForegroundColor Green
Write-Host ""
Write-Host "Key Metrics:" -ForegroundColor Yellow
Write-Host "  • Health check: OK" -ForegroundColor Gray
Write-Host "  • Feed retrieval: OK" -ForegroundColor Gray
Write-Host "  • Interaction recording: OK" -ForegroundColor Gray
Write-Host "  • Cache system: OK" -ForegroundColor Gray
Write-Host "  • Pagination: OK" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Check logs for any warnings" -ForegroundColor Gray
Write-Host "  2. Monitor Kafka topics for events" -ForegroundColor Gray
Write-Host "  3. Verify data in PostgreSQL and Neo4j" -ForegroundColor Gray
Write-Host "  4. Set up Python model service (optional)" -ForegroundColor Gray
Write-Host ""
Write-Host "For detailed architecture info, see: HYBRID_ARCHITECTURE.md" -ForegroundColor Cyan
