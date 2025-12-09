# Test Script for Recommendation System Fixes
# December 9, 2025

$userId = "31ba8a23-8a4e-4b24-99c2-0d768e617e71"

Write-Host "`n╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Recommendation System - Fix Verification Test        ║" -ForegroundColor Cyan
Write-Host "║  December 9, 2025                                     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# Test 1: Service Health
Write-Host "`n[Test 1/5] Checking Service Health..." -ForegroundColor Yellow
try {
    $rHealth = Invoke-RestMethod -Uri "http://localhost:8095/actuator/health" -Method Get
    $pHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    
    if ($rHealth.status -eq "UP" -and $pHealth.status -eq "healthy") {
        Write-Host "✅ PASS: Services are healthy" -ForegroundColor Green
    } else {
        Write-Host "❌ FAIL: Services not healthy" -ForegroundColor Red
        Write-Host "  Recommend-service: $($rHealth.status)" -ForegroundColor Yellow
        Write-Host "  Python model: $($pHealth.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ FAIL: Cannot connect to services" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
    exit 1
}

# Test 2: Bean Definition (Check logs)
Write-Host "`n[Test 2/5] Checking for Bean Definition Errors..." -ForegroundColor Yellow
$beanErrors = docker-compose logs recommend-service 2>&1 | Select-String "BeanDefinitionOverrideException"
if ($beanErrors.Count -eq 0) {
    Write-Host "✅ PASS: No bean definition errors" -ForegroundColor Green
} else {
    Write-Host "❌ FAIL: Bean definition errors found" -ForegroundColor Red
    Write-Host $beanErrors -ForegroundColor Yellow
}

# Test 3: Recommendation Scores Diversity
Write-Host "`n[Test 3/5] Checking Recommendation Score Diversity..." -ForegroundColor Yellow
try {
    $feedResponse = Invoke-RestMethod -Uri "http://localhost:8095/api/recommendations/feed?userId=$userId&page=0&size=5" -Method Get
    $posts = $feedResponse.content
    
    if ($posts.Count -eq 0) {
        Write-Host "⚠️  WARNING: No posts returned" -ForegroundColor Yellow
    } else {
        $scores = $posts | ForEach-Object { $_.score }
        $uniqueScores = $scores | Select-Object -Unique
        
        Write-Host "  Posts returned: $($posts.Count)" -ForegroundColor Cyan
        Write-Host "  Unique scores: $($uniqueScores.Count)" -ForegroundColor Cyan
        Write-Host "  Score range: $([math]::Round($scores | Measure-Object -Minimum).Minimum, 4) - $([math]::Round($scores | Measure-Object -Maximum).Maximum, 4)" -ForegroundColor Cyan
        
        # Display scores
        Write-Host "`n  Top Posts:" -ForegroundColor Cyan
        $posts | Select-Object -First 5 | ForEach-Object {
            Write-Host "    [$($_.postId)] → score: $([math]::Round($_.score, 4))" -ForegroundColor White
        }
        
        # Check for diversity
        if ($uniqueScores.Count -eq 1 -and $uniqueScores[0] -in @(0.0, 0.3)) {
            Write-Host "`n❌ FAIL: All scores are identical ($($uniqueScores[0]))" -ForegroundColor Red
            Write-Host "  This indicates the ML model is not working properly" -ForegroundColor Yellow
        } elseif ($uniqueScores.Count -lt ($posts.Count * 0.5)) {
            Write-Host "`n⚠️  WARNING: Low score diversity (only $($uniqueScores.Count) unique out of $($posts.Count))" -ForegroundColor Yellow
        } else {
            Write-Host "`n✅ PASS: Scores are diverse" -ForegroundColor Green
        }
    }
} catch {
    Write-Host "❌ FAIL: Cannot get recommendations" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Test 4: User Action Event Flow (Check recent logs)
Write-Host "`n[Test 4/5] Checking User Action Event Flow..." -ForegroundColor Yellow

$publishedEvents = docker-compose logs --tail=50 post-service 2>&1 | Select-String "Published user_action event"
$receivedEvents = docker-compose logs --tail=50 recommend-service 2>&1 | Select-String "Received user_action"
$savedFeedback = docker-compose logs --tail=50 recommend-service 2>&1 | Select-String "Saved user feedback"

Write-Host "  Published events (post-service): $($publishedEvents.Count)" -ForegroundColor Cyan
Write-Host "  Received events (recommend-service): $($receivedEvents.Count)" -ForegroundColor Cyan
Write-Host "  Saved feedback: $($savedFeedback.Count)" -ForegroundColor Cyan

if ($publishedEvents.Count -gt 0 -and $receivedEvents.Count -gt 0 -and $savedFeedback.Count -gt 0) {
    Write-Host "`n✅ PASS: User action events are flowing correctly" -ForegroundColor Green
    Write-Host "  Latest event:" -ForegroundColor Cyan
    Write-Host "    $($publishedEvents[-1])" -ForegroundColor White
} elseif ($publishedEvents.Count -gt 0 -and $receivedEvents.Count -eq 0) {
    Write-Host "`n❌ FAIL: Events published but not received" -ForegroundColor Red
    Write-Host "  Check Kafka consumer configuration" -ForegroundColor Yellow
} else {
    Write-Host "`n⚠️  INFO: No recent user actions detected" -ForegroundColor Yellow
    Write-Host "  Try liking a post in the UI and run this test again" -ForegroundColor Cyan
}

# Test 5: Python Model Errors
Write-Host "`n[Test 5/5] Checking for Python Model Errors..." -ForegroundColor Yellow
$pythonErrors = docker-compose logs --tail=100 recommend-service 2>&1 | Select-String "ERROR.*NoneType|ERROR.*unsupported operand"

if ($pythonErrors.Count -eq 0) {
    Write-Host "✅ PASS: No NoneType multiplication errors" -ForegroundColor Green
} else {
    Write-Host "❌ FAIL: Python model errors detected" -ForegroundColor Red
    Write-Host "  Recent errors:" -ForegroundColor Yellow
    $pythonErrors | Select-Object -First 3 | ForEach-Object {
        Write-Host "    $_" -ForegroundColor White
    }
}

# Summary
Write-Host "`n╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Test Summary                                          ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "  1. If all tests PASS → System is working correctly ✅" -ForegroundColor Green
Write-Host "  2. If Test 4 shows no events → Like a post in UI and rerun" -ForegroundColor Cyan
Write-Host "  3. If scores are all identical → Check FINAL-FIX-SUMMARY-DEC-9.md" -ForegroundColor Yellow
Write-Host "  4. If bean errors → Verify KafkaConfig.java changes" -ForegroundColor Yellow

Write-Host "`nDetailed logs:" -ForegroundColor Cyan
Write-Host "  docker-compose logs recommend-service | Select-String 'ERROR'" -ForegroundColor White
Write-Host "  docker-compose logs post-service | Select-String 'user_action'" -ForegroundColor White

Write-Host "`n" -ForegroundColor White
