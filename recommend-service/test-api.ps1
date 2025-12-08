# Script to test Recommend Service APIs
# Tests both Python ML service and Java API service

param(
    [string]$PythonUrl = "http://localhost:8000",
    [string]$JavaUrl = "http://localhost:8095",
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# Colors
function Write-Success { param($msg) Write-Host "✓ $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "✗ $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "ℹ $msg" -ForegroundColor Cyan }
function Write-Test { param($msg) Write-Host "🧪 $msg" -ForegroundColor Yellow }

Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Recommend Service API Test Suite                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$testResults = @{
    Passed = 0
    Failed = 0
    Skipped = 0
}

# Helper function to make HTTP requests
function Invoke-ApiTest {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Method = "GET",
        [object]$Body = $null,
        [int]$ExpectedStatus = 200
    )
    
    Write-Test "Testing: $Name"
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            ContentType = "application/json"
            TimeoutSec = 30
        }
        
        if ($Body) {
            $params.Body = ($Body | ConvertTo-Json -Depth 10)
        }
        
        if ($Verbose) {
            Write-Info "  URL: $Url"
            Write-Info "  Method: $Method"
            if ($Body) {
                Write-Info "  Body: $($params.Body)"
            }
        }
        
        $response = Invoke-RestMethod @params
        
        Write-Success "$Name - Status: 200 OK"
        if ($Verbose -and $response) {
            Write-Info "  Response: $($response | ConvertTo-Json -Depth 2 -Compress)"
        }
        
        $script:testResults.Passed++
        return @{ Success = $true; Data = $response }
        
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq $ExpectedStatus) {
            Write-Success "$Name - Expected status: $statusCode"
            $script:testResults.Passed++
            return @{ Success = $true }
        } else {
            Write-Error "$Name - Failed: $($_.Exception.Message)"
            if ($Verbose) {
                Write-Error "  Details: $($_.ErrorDetails.Message)"
            }
            $script:testResults.Failed++
            return @{ Success = $false; Error = $_.Exception.Message }
        }
    }
}

# ============================================================
# PART 1: Test Python ML Service
# ============================================================

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host "  PART 1: Testing Python ML Service ($PythonUrl)" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow

# Test 1.1: Health Check
$result = Invoke-ApiTest -Name "Health Check" -Url "$PythonUrl/health"
Start-Sleep -Milliseconds 500

# Test 1.2: Root endpoint
$result = Invoke-ApiTest -Name "Root Endpoint" -Url "$PythonUrl/"
Start-Sleep -Milliseconds 500

# Test 1.3: Embed single post
$postData = @{
    post_id = "post_001"
    content = "Thông báo học bổng VIED 2024 dành cho sinh viên năm 3 trở lên"
    tags = @("scholarship", "announcement")
}
$result = Invoke-ApiTest -Name "Embed Single Post" -Url "$PythonUrl/embed/post" -Method "POST" -Body $postData
Start-Sleep -Milliseconds 500

# Test 1.4: Embed batch posts
$batchData = @{
    posts = @(
        @{
            post_id = "post_001"
            content = "Hội thảo khoa học AI và Machine Learning"
            tags = @("event", "research")
        },
        @{
            post_id = "post_002"
            content = "Tuyển sinh đại học 2024"
            tags = @("announcement")
        }
    )
}
$result = Invoke-ApiTest -Name "Embed Batch Posts" -Url "$PythonUrl/embed/post/batch" -Method "POST" -Body $batchData
Start-Sleep -Milliseconds 500

# Test 1.5: Embed user profile
$userData = @{
    user_id = "user_001"
    interests = @("machine learning", "data science", "python")
    major = "Computer Science"
    bio = "Sinh viên năm 4 chuyên ngành AI"
}
$result = Invoke-ApiTest -Name "Embed User Profile" -Url "$PythonUrl/embed/user" -Method "POST" -Body $userData
Start-Sleep -Milliseconds 500

# Test 1.6: Compute similarity
$similarityData = @{
    embedding1 = @(0.1, 0.2, 0.3, 0.4, 0.5)
    embedding2 = @(0.15, 0.25, 0.35, 0.45, 0.55)
}
$result = Invoke-ApiTest -Name "Compute Similarity" -Url "$PythonUrl/similarity" -Method "POST" -Body $similarityData
Start-Sleep -Milliseconds 500

# ============================================================
# PART 2: Test Java Recommendation Service
# ============================================================

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host "  PART 2: Testing Java Recommendation Service ($JavaUrl)" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow

# Test 2.1: Health Check
Write-Host ""
Write-Host "Testing Java service health..." -ForegroundColor Gray
try {
    $result = Invoke-ApiTest -Name "Java Health Check" -Url "$JavaUrl/actuator/health"
} catch {
    Write-Host "  Actuator health failed, trying API endpoint..." -ForegroundColor Yellow
    $result = Invoke-ApiTest -Name "Java API Endpoint Check" -Url "$JavaUrl/api/recommendation/feed?userId=health&size=1"
}
Start-Sleep -Milliseconds 500

# Test 2.2: Get personalized feed (hybrid architecture)
$result = Invoke-ApiTest -Name "Get Personalized Feed" -Url "$JavaUrl/api/recommendation/feed?userId=user_001&page=0&size=20"
Start-Sleep -Milliseconds 500

# Test 2.3: Get recommendations (GET)
$result = Invoke-ApiTest -Name "Get Recommendations (GET)" -Url "$JavaUrl/api/recommend/posts?userId=user_001&page=0&size=10"
Start-Sleep -Milliseconds 500

# Test 2.4: Get recommendations (POST)
$recommendRequest = @{
    userId = "user_001"
    page = 0
    size = 20
    includeExplanations = $true
    filters = @{
        categories = @("research", "scholarship")
        minScore = 0.5
    }
}
$result = Invoke-ApiTest -Name "Get Recommendations (POST)" -Url "$JavaUrl/api/recommend/posts" -Method "POST" -Body $recommendRequest
Start-Sleep -Milliseconds 500

# Test 2.5: Record feedback
$feedbackData = @{
    userId = "user_001"
    postId = "post_001"
    feedbackType = "LIKE"
}
$result = Invoke-ApiTest -Name "Record Feedback" -Url "$JavaUrl/api/recommend/feedback" -Method "POST" -Body $feedbackData
Start-Sleep -Milliseconds 500

# Test 2.6: Record interaction
$interactionData = @{
    userId = "user_001"
    postId = "post_001"
    type = "VIEW"
    viewDuration = 45
    context = @{
        source = "feed"
        position = 3
    }
}
$result = Invoke-ApiTest -Name "Record Interaction" -Url "$JavaUrl/api/recommendation/interaction" -Method "POST" -Body $interactionData
Start-Sleep -Milliseconds 500

# Test 2.7: Invalidate cache
$cacheData = @{
    userId = "user_001"
}
$result = Invoke-ApiTest -Name "Invalidate Cache" -Url "$JavaUrl/api/recommendation/cache/invalidate" -Method "POST" -Body $cacheData
Start-Sleep -Milliseconds 500

# ============================================================
# Test Summary
# ============================================================

Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Test Results Summary                             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Total Tests: $($testResults.Passed + $testResults.Failed)" -ForegroundColor White
Write-Success "Passed: $($testResults.Passed)"
if ($testResults.Failed -gt 0) {
    Write-Error "Failed: $($testResults.Failed)"
} else {
    Write-Host "  Failed: $($testResults.Failed)" -ForegroundColor Gray
}
Write-Host ""

if ($testResults.Failed -eq 0) {
    Write-Host "🎉 All tests passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "⚠️  Some tests failed. Check the output above for details." -ForegroundColor Yellow
    exit 1
}
