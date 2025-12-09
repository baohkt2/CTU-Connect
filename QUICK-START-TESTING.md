# QUICK START - Testing Guide

## Prerequisites Check

```powershell
# Check all services are running
docker ps | Select-String "post-service|recommend-service|postgres|mongodb|redis|kafka"

# Check Python model service (runs separately)
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

## Run Automated Test

```powershell
.\test-recommendation-system.ps1 -UserId "31ba8a23-8a4e-4b24-99c2-0d768e617e71"
```

## Common Issues & Quick Fixes

### Issue 1: 422 Unprocessable Entity

**Check logs** (now with detailed debugging):
```powershell
# Recommend service - shows exact request being sent
Get-Content ".\recommend-service\java-api\logs\recommend-service.log" -Tail 50 | Select-String "Calling Python|Sample post|422"

# Python service - shows what it received
Get-Content ".\recommend-service\python-model\logs\app.log" -Tail 50 | Select-String "Prediction request|Sample post|Validation error"
```

**Likely causes**:
- Field type mismatch (e.g., null where int expected)
- Missing required field in CandidatePost
- Field naming mismatch (camelCase vs snake_case)

### Issue 2: All Posts Same Score (0.3000)

**Symptom**: All recommendations have identical scores

**Cause**: Fallback ranking triggered (Python service failed)

**Check**:
```powershell
# 1. Is Python service running?
curl http://localhost:8000/health

# 2. Check if it's being called
Get-Content ".\recommend-service\java-api\logs\recommend-service.log" -Tail 20 | Select-String "Calling Python"

# 3. Check for errors
Get-Content ".\recommend-service\python-model\logs\app.log" -Tail 20
```

### Issue 3: User Interactions Not Saved

**Check Kafka flow**:
```bash
# Monitor user_action topic in real-time
docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic user_action

# In another terminal, trigger an interaction (like a post)
# You should see the event appear
```

**Check database**:
```bash
docker exec -it postgres psql -U postgres -d recommend_db \
  -c "SELECT COUNT(*) FROM user_feedback;"
```

**Check logs**:
```powershell
# post-service publishing
Get-Content ".\post-service\target\*.log" -Tail 20 -ErrorAction SilentlyContinue | Select-String "Published user_action"

# recommend-service consuming
Get-Content ".\recommend-service\java-api\logs\recommend-service.log" -Tail 20 | Select-String "Processing user_action"
```

## Manual Testing Steps

### 1. Test Feed Generation
```powershell
$userId = "31ba8a23-8a4e-4b24-99c2-0d768e617e71"
Invoke-RestMethod -Uri "http://localhost:8095/api/recommendations/feed?userId=$userId&size=10"
```

### 2. Test Python Service Directly
```powershell
$body = @{
    userAcademic = @{
        userId = "test-user"
        major = "Công nghệ thông tin"
        faculty = "CNTT"
        degree = "Đại học"
        batch = "K47"
    }
    userHistory = @()
    candidatePosts = @(
        @{
            postId = "test-1"
            content = "Test post content"
            hashtags = @()
            likeCount = 10
            commentCount = 5
            shareCount = 2
            viewCount = 100
        }
    )
    topK = 10
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:8000/api/model/predict" -Method Post -Body $body -ContentType "application/json"
```

### 3. Check Database State
```bash
# User feedback
docker exec -it postgres psql -U postgres -d recommend_db \
  -c "SELECT user_id, post_id, feedback_type, timestamp FROM user_feedback ORDER BY timestamp DESC LIMIT 5;"

# Post embeddings
docker exec -it postgres psql -U postgres -d recommend_db \
  -c "SELECT post_id, like_count, score FROM post_embeddings LIMIT 5;"
```

## Monitor in Real-Time

```powershell
# Terminal 1: Recommend service
Get-Content ".\recommend-service\java-api\logs\recommend-service.log" -Wait -Tail 10

# Terminal 2: Python service  
Get-Content ".\recommend-service\python-model\logs\app.log" -Wait -Tail 10

# Terminal 3: Kafka events
docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic user_action
```

## Expected Results

### ✅ Successful Flow
- Python service health check passes
- Feed returns 10+ posts with varying scores
- Scores range from 0.2 to 1.0 (not all the same)
- Second request faster (cache hit)
- User interactions appear in Kafka immediately
- Database updated within 1-2 seconds

### ⚠️ Degraded Mode (Fallback)
- Python service unavailable
- Posts ranked by popularity only
- All scores may be similar (0.2-0.5)
- Still functional, just not personalized

### ❌ Broken States
- 422 errors from Python → Request format mismatch
- Empty feed → No posts in database
- No interaction recording → Kafka/database issue

## Debug Commands Reference

```powershell
# Quick health check all services
curl http://localhost:8000/health  # Python
curl http://localhost:8095/actuator/health  # Recommend
curl http://localhost:8085/actuator/health  # Post

# Check recent errors in recommend service
Get-Content ".\recommend-service\java-api\logs\recommend-service.log" | Select-String "ERROR|422" | Select-Object -Last 10

# Check Python service errors  
Get-Content ".\recommend-service\python-model\logs\app.log" | Select-String "ERROR" | Select-Object -Last 10

# Count events in Kafka
docker exec -it kafka kafka-run-class kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 \
  --topic user_action

# Check Redis cache
docker exec -it redis redis-cli KEYS "recommendation:*"
```

## What to Do If Tests Fail

1. **Python 422 error**: Check new detailed logs in both Java and Python
2. **Empty feed**: Verify database has posts and user has profile
3. **Same scores**: Check if Python service is actually being called
4. **No interactions recorded**: Check Kafka consumer logs for errors
5. **Services not responding**: Restart services and check docker logs

## Next Steps

After confirming tests pass:
1. Test with multiple users
2. Test interaction flow (like → see updated feed)
3. Test new user flow (cold start)
4. Monitor performance over time
5. Check cache hit rates in logs
