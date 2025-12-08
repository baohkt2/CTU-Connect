# Testing Guide - Recommend Service

## Overview

This guide covers testing the Recommend Service, which consists of two components:
- **Python ML Service** (port 8000): PhoBERT embeddings and ML models
- **Java API Service** (port 8095): Business logic and orchestration

## Test Scripts

### 1. Quick Health Check

**Simple test to verify both services are running:**

```powershell
cd recommend-service
.\test-api-simple.ps1
```

**What it tests:**
- Python service health
- Java service health
- Basic embedding functionality
- Basic recommendation API

**Expected output:**
```
✓ Python service is running
✓ Java service is running
✓ Embedding API works
✓ Recommendation API works
```

### 2. Comprehensive API Tests

**Full test suite covering all endpoints:**

```powershell
.\test-api.ps1
```

**With verbose output:**
```powershell
.\test-api.ps1 -Verbose
```

**Custom URLs:**
```powershell
.\test-api.ps1 -PythonUrl "http://localhost:8000" -JavaUrl "http://localhost:8095"
```

**What it tests:**

**Python ML Service:**
- Health check
- Single post embedding
- Batch post embedding
- User profile embedding
- Similarity computation
- Batch similarity computation

**Java API Service:**
- Health check
- Personalized feed (hybrid architecture)
- GET recommendations
- POST recommendations with filters
- Feedback recording
- Interaction tracking
- Cache invalidation

### 3. Scenario Testing

**Real-world workflow testing:**

```powershell
.\test-scenarios.ps1
```

**What it tests:**

**Scenario 1: New User First Visit**
- New user requests feed
- User views posts
- Interactions are recorded

**Scenario 2: Active User with History**
- Multiple view interactions
- Like/dislike feedback
- Feed refresh after interactions

**Scenario 3: Academic Content Discovery**
- Filtered recommendations
- Category-based search
- Explanation generation

**Scenario 4: Batch Processing**
- Multiple posts embedding
- Batch operations

**Scenario 5: Cache Management**
- Cache hits
- Cache invalidation
- Fresh recommendations

**Scenario 6: Edge Cases**
- Empty content
- Very long content
- Invalid inputs
- Pagination limits

**Scenario 7: Performance**
- Concurrent requests
- Response times

## Manual Testing

### Using Swagger UI

**Python Service:**
Open http://localhost:8000/docs

**Java Service:**
Open http://localhost:8095/swagger-ui.html (if configured)

### Using curl

**Get Python health:**
```bash
curl http://localhost:8000/health
```

**Embed a post:**
```bash
curl -X POST http://localhost:8000/embed/post \
  -H "Content-Type: application/json" \
  -d '{
    "post_id": "test_001",
    "content": "Thông báo học bổng VIED 2024",
    "tags": ["scholarship"]
  }'
```

**Get recommendations:**
```bash
curl "http://localhost:8095/api/recommendation/feed?userId=user_001&size=10"
```

**Record interaction:**
```bash
curl -X POST http://localhost:8095/api/recommendation/interaction \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user_001",
    "postId": "post_001",
    "type": "VIEW",
    "viewDuration": 45
  }'
```

### Using PowerShell

**Get recommendations:**
```powershell
$response = Invoke-RestMethod -Uri "http://localhost:8095/api/recommendation/feed?userId=user_001&size=10"
$response | ConvertTo-Json -Depth 3
```

**Embed multiple posts:**
```powershell
$body = @{
    posts = @(
        @{
            post_id = "p1"
            content = "AI research"
            tags = @("research", "ai")
        },
        @{
            post_id = "p2"
            content = "Scholarship announcement"
            tags = @("scholarship")
        }
    )
} | ConvertTo-Json -Depth 10

$response = Invoke-RestMethod -Uri "http://localhost:8000/embed/post/batch" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response | ConvertTo-Json -Depth 3
```

## Test Data Setup

### Create Test Users

```sql
-- Neo4j
CREATE (u:User {userId: 'test_user_001', name: 'Test User', major: 'CS'})
CREATE (u:User {userId: 'test_user_002', name: 'Test User 2', major: 'AI'})
```

### Create Test Posts

```sql
-- PostgreSQL
INSERT INTO posts (post_id, content, category, created_at) VALUES
('test_post_001', 'AI research conference', 'research', NOW()),
('test_post_002', 'Scholarship opportunity', 'scholarship', NOW());
```

## Performance Testing

### Load Testing with Apache Bench

```bash
# Test Python embedding endpoint
ab -n 1000 -c 10 -p post_data.json -T application/json http://localhost:8000/embed/post

# Test Java recommendation endpoint
ab -n 1000 -c 10 http://localhost:8095/api/recommendation/feed?userId=test&size=10
```

### Stress Testing

```powershell
# Concurrent requests
$jobs = @()
for ($i = 0; $i -lt 100; $i++) {
    $jobs += Start-Job -ScriptBlock {
        Invoke-RestMethod "http://localhost:8095/api/recommendation/feed?userId=user_$($using:i)"
    }
}
$jobs | Wait-Job | Receive-Job
$jobs | Remove-Job
```

## Troubleshooting Tests

### Python Service Not Responding

```powershell
# Check if running
Get-Process | Where-Object {$_.ProcessName -like "*python*"}

# Check port
netstat -ano | findstr :8000

# Restart service
cd recommend-service\python-model
.\run-dev.ps1
```

### Java Service Not Responding

```powershell
# Check if running
Get-Process | Where-Object {$_.ProcessName -like "*java*"}

# Check port
netstat -ano | findstr :8095

# Check logs
tail -f recommend-service/java-api/logs/application.log
```

### Database Connection Issues

```powershell
# Check PostgreSQL
docker exec -it ctu-recommend-postgres psql -U recommend_user -d recommend_db -c "SELECT 1;"

# Check Redis
docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass ping

# Check Neo4j
curl http://localhost:7474
```

### Model Loading Issues

```powershell
# Check if model exists
Test-Path "recommend-service\python-model\model\academic_posts_model\config.json"

# Use HuggingFace model instead
cd recommend-service\python-model
.\fix-model-config.ps1
# Choose option 2
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Test Recommend Service
  run: |
    cd recommend-service
    ./test-api-simple.ps1
    ./test-api.ps1
    ./test-scenarios.ps1
```

### Jenkins Pipeline Example

```groovy
stage('Test Recommend Service') {
    steps {
        powershell '''
            cd recommend-service
            .\\test-api.ps1 -Verbose
        '''
    }
}
```

## Best Practices

1. **Always test both services** - Python and Java work together
2. **Test with realistic data** - Use Vietnamese content for embeddings
3. **Check response times** - Embeddings should be < 1s, recommendations < 2s
4. **Monitor logs** - Check for errors even if tests pass
5. **Test edge cases** - Empty content, very long text, special characters
6. **Verify cache behavior** - Test with and without cache
7. **Test concurrency** - Ensure thread-safe operations

## Metrics to Monitor

- **Response time**: < 2s for recommendations
- **Embedding time**: < 1s per post
- **Cache hit rate**: > 70% after warmup
- **Error rate**: < 1%
- **Concurrent users**: Support 100+ simultaneous requests

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review API documentation at `/docs`
- Check database connections
- Verify model files exist
