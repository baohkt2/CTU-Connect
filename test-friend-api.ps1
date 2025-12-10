# Test Friend Feature APIs
# This script tests all friend-related API endpoints

$API_BASE = "http://localhost:8080/api/users"
$TOKEN = "" # You need to set this with a valid JWT token

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Friend Feature API Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($TOKEN -eq "") {
    Write-Host "ERROR: Please set TOKEN variable with a valid JWT token" -ForegroundColor Red
    Write-Host "Example: `$TOKEN = 'your-jwt-token-here'" -ForegroundColor Yellow
    exit 1
}

$headers = @{
    "Authorization" = "Bearer $TOKEN"
    "Content-Type" = "application/json"
}

function Test-Endpoint {
    param(
        [string]$Method,
        [string]$Endpoint,
        [string]$Description,
        [object]$Body = $null
    )
    
    Write-Host "Testing: $Description" -ForegroundColor Yellow
    Write-Host "  $Method $Endpoint" -ForegroundColor Gray
    
    try {
        if ($Body) {
            $response = Invoke-RestMethod -Uri "$API_BASE$Endpoint" -Method $Method -Headers $headers -Body ($Body | ConvertTo-Json)
        } else {
            $response = Invoke-RestMethod -Uri "$API_BASE$Endpoint" -Method $Method -Headers $headers
        }
        
        Write-Host "  ✓ SUCCESS" -ForegroundColor Green
        Write-Host "  Response: $($response | ConvertTo-Json -Compress -Depth 2)" -ForegroundColor Gray
        return $true
    } catch {
        Write-Host "  ✗ FAILED" -ForegroundColor Red
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    Write-Host ""
}

# Test results tracker
$results = @{
    passed = 0
    failed = 0
    total = 0
}

function Record-Result {
    param([bool]$success)
    $results.total++
    if ($success) { $results.passed++ } else { $results.failed++ }
}

Write-Host "1. Friend List Management" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Record-Result (Test-Endpoint "GET" "/me/friends" "Get my friends list")
Record-Result (Test-Endpoint "GET" "/me/friends?page=0&size=5" "Get my friends list (paginated)")

Write-Host "`n2. Friend Requests Management" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Record-Result (Test-Endpoint "GET" "/me/friend-requests" "Get received friend requests")
Record-Result (Test-Endpoint "GET" "/me/friend-requested" "Get sent friend requests")

Write-Host "`n3. Friend Suggestions" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Record-Result (Test-Endpoint "GET" "/friend-suggestions?limit=10" "Get friend suggestions (basic)")
Record-Result (Test-Endpoint "GET" "/friend-suggestions/search?limit=10" "Get friend suggestions (no filters)")
Record-Result (Test-Endpoint "GET" "/friend-suggestions/search?faculty=Công nghệ thông tin&limit=10" "Search by faculty")
Record-Result (Test-Endpoint "GET" "/friend-suggestions/search?batch=2020&limit=10" "Search by batch")
Record-Result (Test-Endpoint "GET" "/friend-suggestions/search?query=nguyen&limit=10" "Search by query")

Write-Host "`n4. Friendship Status (requires target user ID)" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Skipping - requires specific target user ID" -ForegroundColor Gray
# Uncomment and set TARGET_USER_ID to test
# $TARGET_USER_ID = "some-user-id"
# Record-Result (Test-Endpoint "GET" "/$TARGET_USER_ID/friendship-status" "Get friendship status")
# Record-Result (Test-Endpoint "GET" "/$TARGET_USER_ID/mutual-friends-count" "Get mutual friends count")
# Record-Result (Test-Endpoint "GET" "/$TARGET_USER_ID/mutual-friends?page=0&size=5" "Get mutual friends list")

Write-Host "`n5. Friend Actions (requires target user ID)" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Skipping - requires specific target user ID" -ForegroundColor Gray
# Uncomment and set TARGET_USER_ID to test
# Record-Result (Test-Endpoint "POST" "/me/invite/$TARGET_USER_ID" "Send friend request")
# Record-Result (Test-Endpoint "POST" "/me/accept-invite/$TARGET_USER_ID" "Accept friend request")
# Record-Result (Test-Endpoint "POST" "/me/reject-invite/$TARGET_USER_ID" "Reject friend request")
# Record-Result (Test-Endpoint "DELETE" "/me/friends/$TARGET_USER_ID" "Remove friend")

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total Tests: $($results.total)" -ForegroundColor White
Write-Host "Passed: $($results.passed)" -ForegroundColor Green
Write-Host "Failed: $($results.failed)" -ForegroundColor Red

if ($results.failed -eq 0) {
    Write-Host "`n✓ All tests passed!" -ForegroundColor Green
} else {
    Write-Host "`n✗ Some tests failed. Please check the errors above." -ForegroundColor Red
}

Write-Host "`nNote: To test friend actions, set `$TARGET_USER_ID and uncomment the relevant sections." -ForegroundColor Yellow
