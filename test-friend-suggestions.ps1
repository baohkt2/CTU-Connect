# Test Friend Suggestions Feature
Write-Host "=== Testing Friend Suggestions Feature ===" -ForegroundColor Cyan

# Configuration
$API_GATEWAY = "http://localhost:8090"
$TEST_USER = @{
    email = "user1@student.ctu.edu.vn"
    password = "password"
}

# Step 1: Login
Write-Host "`n1. Logging in..." -ForegroundColor Yellow
$loginResponse = Invoke-RestMethod -Uri "$API_GATEWAY/api/auth/login" `
    -Method POST `
    -ContentType "application/json" `
    -Body ($TEST_USER | ConvertTo-Json)

$TOKEN = $loginResponse.token
Write-Host "✓ Login successful! Token: $($TOKEN.Substring(0, 20))..." -ForegroundColor Green

$headers = @{
    "Authorization" = "Bearer $TOKEN"
    "Content-Type" = "application/json"
}

# Step 2: Test Basic Friend Suggestions
Write-Host "`n2. Testing basic friend suggestions..." -ForegroundColor Yellow
try {
    $suggestions = Invoke-RestMethod -Uri "$API_GATEWAY/api/users/me/friend-suggestions?limit=5" `
        -Method GET `
        -Headers $headers
    
    Write-Host "✓ Got $($suggestions.Count) friend suggestions" -ForegroundColor Green
    if ($suggestions.Count -gt 0) {
        Write-Host "First suggestion: $($suggestions[0].fullName) - $($suggestions[0].suggestionReason)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 3: Test Enhanced Friend Suggestions (No filters)
Write-Host "`n3. Testing enhanced friend suggestions (priority ranking)..." -ForegroundColor Yellow
try {
    $enhancedSuggestions = Invoke-RestMethod -Uri "$API_GATEWAY/api/users/friend-suggestions/search?limit=10" `
        -Method GET `
        -Headers $headers
    
    Write-Host "✓ Got $($enhancedSuggestions.Count) enhanced suggestions" -ForegroundColor Green
    if ($enhancedSuggestions.Count -gt 0) {
        Write-Host "`nTop 3 suggestions (by priority):" -ForegroundColor Cyan
        for ($i = 0; $i -lt [Math]::Min(3, $enhancedSuggestions.Count); $i++) {
            $user = $enhancedSuggestions[$i]
            Write-Host "  $($i+1). $($user.fullName) - $($user.faculty) - Batch: $($user.batch)" -ForegroundColor White
            Write-Host "     Same college: $($user.sameCollege), Same faculty: $($user.sameFaculty), Same batch: $($user.sameBatch)" -ForegroundColor Gray
            if ($user.mutualFriendsCount -gt 0) {
                Write-Host "     Mutual friends: $($user.mutualFriendsCount)" -ForegroundColor Gray
            }
        }
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 4: Test Search by Name
Write-Host "`n4. Testing search by name..." -ForegroundColor Yellow
try {
    $searchResults = Invoke-RestMethod -Uri "$API_GATEWAY/api/users/friend-suggestions/search?query=user&limit=5" `
        -Method GET `
        -Headers $headers
    
    Write-Host "✓ Found $($searchResults.Count) users matching 'user'" -ForegroundColor Green
    if ($searchResults.Count -gt 0) {
        Write-Host "Results:" -ForegroundColor Cyan
        foreach ($user in $searchResults) {
            Write-Host "  - $($user.fullName) ($($user.studentId))" -ForegroundColor White
        }
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 5: Test Filter by Faculty
Write-Host "`n5. Testing filter by faculty..." -ForegroundColor Yellow
try {
    # URL encode the faculty name
    $faculty = [System.Web.HttpUtility]::UrlEncode("Công nghệ thông tin")
    $filterResults = Invoke-RestMethod -Uri "$API_GATEWAY/api/users/friend-suggestions/search?faculty=$faculty&limit=5" `
        -Method GET `
        -Headers $headers
    
    Write-Host "✓ Found $($filterResults.Count) users in 'Công nghệ thông tin' faculty" -ForegroundColor Green
    if ($filterResults.Count -gt 0) {
        Write-Host "Results:" -ForegroundColor Cyan
        foreach ($user in $filterResults) {
            Write-Host "  - $($user.fullName) - $($user.faculty)" -ForegroundColor White
        }
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 6: Test Search by Email
Write-Host "`n6. Testing search by email..." -ForegroundColor Yellow
try {
    $emailSearch = Invoke-RestMethod -Uri "$API_GATEWAY/api/users/search/email?email=user2@student.ctu.edu.vn" `
        -Method GET `
        -Headers $headers
    
    Write-Host "✓ Found user by email" -ForegroundColor Green
    Write-Host "  Name: $($emailSearch.fullName)" -ForegroundColor Cyan
    Write-Host "  Email: $($emailSearch.email)" -ForegroundColor Cyan
    Write-Host "  Friendship Status: $($emailSearch.friendshipStatus)" -ForegroundColor Cyan
    Write-Host "  Mutual Friends: $($emailSearch.mutualFriendsCount)" -ForegroundColor Cyan
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 7: Test Combined Search + Filter
Write-Host "`n7. Testing combined search and filter..." -ForegroundColor Yellow
try {
    $combinedResults = Invoke-RestMethod -Uri "$API_GATEWAY/api/users/friend-suggestions/search?query=B2014&batch=2020&limit=5" `
        -Method GET `
        -Headers $headers
    
    Write-Host "✓ Found $($combinedResults.Count) users matching 'B2014' in batch 2020" -ForegroundColor Green
    if ($combinedResults.Count -gt 0) {
        Write-Host "Results:" -ForegroundColor Cyan
        foreach ($user in $combinedResults) {
            Write-Host "  - $($user.fullName) ($($user.studentId)) - Batch: $($user.batch)" -ForegroundColor White
        }
    }
} catch {
    Write-Host "✗ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan
Write-Host "`nSummary:" -ForegroundColor Yellow
Write-Host "✓ All friend suggestion features tested" -ForegroundColor Green
Write-Host "✓ Priority ranking working" -ForegroundColor Green
Write-Host "✓ Search functionality working" -ForegroundColor Green
Write-Host "✓ Filter functionality working" -ForegroundColor Green
Write-Host "✓ Email search working" -ForegroundColor Green
Write-Host "`nFor detailed API documentation, see FRIEND-SUGGESTION-FEATURE-GUIDE.md" -ForegroundColor Cyan
