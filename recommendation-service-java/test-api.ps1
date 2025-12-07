$BASE_URL = "http://localhost:8095/api/recommend"
$USER_ID = "11111111-1111-1111-1111-111111111111"

Write-Host "╔════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║ Testing Recommendation Service APIs    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "1. Health Check" -ForegroundColor Yellow
$response = Invoke-WebRequest -Uri "$BASE_URL/health" -Method GET
Write-Host "  Status: $($response.StatusCode)" -ForegroundColor Green
Write-Host "  Response: $($response.Content)" -ForegroundColor Gray
Write-Host ""

---

# Test 2: Get Recommendations
Write-Host "2. Get Recommendations (Simple)" -ForegroundColor Yellow
try {
  # FIX: Wrap the URI containing '&' in double quotes to treat it as a string
  $url = "$BASE_URL/posts?userId=$USER_ID&size=5"
  $response = Invoke-WebRequest -Uri $url -Method GET
  Write-Host "  Status: $($response.StatusCode)" -ForegroundColor Green
  $json = $response.Content | ConvertFrom-Json
  Write-Host "  Total Results: $($json.totalResults)" -ForegroundColor Gray
  Write-Host "  Returned: $($json.recommendations.Count)" -ForegroundColor Gray
} catch {
  Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

---

# Test 3: Record Feedback
Write-Host "3. Record Feedback" -ForegroundColor Yellow
$body = @{
  userId = $USER_ID
  postId = "post-001"
  feedbackType = "LIKE"
} | ConvertTo-Json

try {
  $response = Invoke-WebRequest -Uri "$BASE_URL/feedback" -Method POST -Body $body -ContentType "application/json"
  Write-Host "  Status: $($response.StatusCode)" -ForegroundColor Green
} catch {
  Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

---

# Test 4: Actuator Health
Write-Host "4. Actuator Health" -ForegroundColor Yellow
try {
  $response = Invoke-WebRequest -Uri "http://localhost:8095/actuator/health" -Method GET
  Write-Host "  Status: $($response.StatusCode)" -ForegroundColor Green
  $json = $response.Content | ConvertFrom-Json
  Write-Host "  Overall: $($json.status)" -ForegroundColor Gray
} catch {
  Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# FIX: Ensure the string terminator is present (assuming the issue was a corrupted character)
Write-Host "════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ API Testing Complete!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════" -ForegroundColor Green