# 🚀 Quick Start Guide - Hybrid Recommendation System

## Prerequisites

- Java 17+
- Maven 3.8+
- Docker & Docker Compose
- (Optional) Python 3.10+ for model service

---

## Step 1: Start Infrastructure (Databases Only)

Since you want to run Java service in IDE and databases in Docker:

```powershell
# Start only the required databases
docker-compose up -d postgres neo4j redis kafka zookeeper
```

Wait for services to be ready (~30 seconds):

```powershell
# Check status
docker-compose ps

# Check logs
docker-compose logs postgres
docker-compose logs neo4j
docker-compose logs redis
```

---

## Step 2: Configure Application

Edit `src/main/resources/application.yml` or use environment variables:

```yaml
# Database connections (localhost since running outside Docker)
spring:
  datasource:
    url: jdbc:postgresql://localhost:5435/recommendation_db
  neo4j:
    uri: bolt://localhost:7687
  data:
    redis:
      host: localhost
      port: 6379
  kafka:
    bootstrap-servers: localhost:9092

# Python Model Service (if available)
recommendation:
  python-service:
    url: http://localhost:8097
    enabled: true  # Set to false if Python service not running
```

---

## Step 3: Run Java Service in IDE

### IntelliJ IDEA:

1. Open `recommendation-service-java` folder as project
2. Wait for Maven dependencies to download
3. Right-click `RecommendationServiceApplication.java`
4. Click "Run 'RecommendationServiceApplication'"

### VS Code:

1. Open folder in VS Code
2. Install "Extension Pack for Java"
3. Press F5 or click "Run" button
4. Select "Spring Boot App"

### Command Line:

```powershell
cd recommendation-service-java
mvn clean install -DskipTests
mvn spring-boot:run
```

The service will start on port **8095** by default.

---

## Step 4: Verify Service is Running

```powershell
# Health check
curl http://localhost:8095/actuator/health

# Expected response:
# {"status":"UP"}
```

---

## Step 5: Load Sample Data

### Option A: Using SQL Script

```powershell
# Connect to PostgreSQL
docker exec -i ctu-connect-postgres psql -U postgres -d recommendation_db < test-data.sql
```

### Option B: Using PowerShell Script

```powershell
.\load-test-data.ps1
```

### Option C: Using Cypher for Neo4j

```powershell
# Load into Neo4j
docker exec -i ctu-connect-neo4j cypher-shell -u neo4j -p password < test-data.cypher
```

---

## Step 6: Test the API

### Get Personalized Feed (Hybrid Mode)

```powershell
# Using curl
curl "http://localhost:8095/api/recommendation/feed?userId=user-001&size=10"

# Using PowerShell Invoke-RestMethod
$response = Invoke-RestMethod -Uri "http://localhost:8095/api/recommendation/feed?userId=user-001&size=10" -Method Get
$response | ConvertTo-Json -Depth 5
```

**Expected Response**:
```json
{
  "userId": "user-001",
  "recommendations": [
    {
      "postId": "post-123",
      "authorId": "user-456",
      "content": "Machine Learning workshop...",
      "score": 0.92,
      "contentSimilarity": 0.85,
      "graphRelationScore": 0.7,
      "academicScore": 0.88,
      "popularityScore": 0.6,
      "academicCategory": "event",
      "createdAt": "2024-01-15T10:30:00"
    }
  ],
  "totalCount": 10,
  "page": 0,
  "size": 10,
  "abVariant": "cached",
  "timestamp": "2024-01-20T15:45:00",
  "processingTimeMs": 45
}
```

### Record User Interaction

```powershell
# Like a post
curl -X POST http://localhost:8095/api/recommendation/interaction `
  -H "Content-Type: application/json" `
  -d '{
    "userId": "user-001",
    "postId": "post-123",
    "type": "LIKE",
    "viewDuration": 5.2,
    "context": {}
  }'
```

### Invalidate Cache (for testing)

```powershell
curl -X POST "http://localhost:8095/api/recommendation/cache/invalidate?userId=user-001"
```

---

## Step 7: Monitor & Debug

### Check Logs

IntelliJ: View in "Run" console
VS Code: View in Terminal
Command Line: Logs appear in console

### Check Database

```powershell
# PostgreSQL
docker exec -it ctu-connect-postgres psql -U postgres -d recommendation_db
# Run: SELECT * FROM post_embeddings LIMIT 5;

# Neo4j
# Open browser: http://localhost:7474
# Run: MATCH (u:User)-[r]->(p:Post) RETURN u, r, p LIMIT 10

# Redis
docker exec -it ctu-connect-redis redis-cli
# Run: KEYS recommend:*
# Run: GET recommend:user-001
```

### Check Kafka

```powershell
# List topics
docker exec ctu-connect-kafka kafka-topics --list --bootstrap-server localhost:9092

# Consume messages
docker exec ctu-connect-kafka kafka-console-consumer --topic user_interaction --from-beginning --bootstrap-server localhost:9092
```

---

## Step 8: (Optional) Run Python Model Service

If you want to use the ML model service:

### Option A: Docker

```powershell
cd recommendation-service
docker build -t ctu-recommendation-python .
docker run -p 8097:8097 ctu-recommendation-python
```

### Option B: Local Python

```powershell
cd recommendation-service
pip install -r requirements.txt
python app.py
```

Verify Python service:
```powershell
curl http://localhost:8097/health
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Java Service (IDE - Port 8095)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GET /api/recommendation/feed                        │  │
│  │  POST /api/recommendation/interaction                │  │
│  │                                                       │  │
│  │  • Orchestration                                     │  │
│  │  • Business Rules                                    │  │
│  │  • Caching (Redis)                                   │  │
│  │  • Event Publishing (Kafka)                          │  │
│  └──────────────────────────────────────────────────────┘  │
└───┬──────────────────┬─────────────────┬──────────────────┬─┘
    │                  │                 │                  │
    ▼                  ▼                 ▼                  ▼
┌─────────┐    ┌──────────────┐  ┌────────────┐  ┌──────────────┐
│PostgreSQL│    │    Neo4j     │  │   Redis    │  │    Kafka     │
│(Docker) │    │  (Docker)    │  │ (Docker)   │  │  (Docker)    │
└─────────┘    └──────────────┘  └────────────┘  └──────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────┐
                                              │ Python Training  │
                                              │    Pipeline      │
                                              │   (Optional)     │
                                              └──────────────────┘
```

---

## Performance Expectations

| Scenario | Response Time | Cache Status |
|----------|---------------|--------------|
| First request (no cache) | 300-500ms | MISS |
| Subsequent requests | 30-50ms | HIT |
| After interaction | 300-500ms | MISS (invalidated) |
| Python service down | 200-300ms | Fallback ranking |

---

## Common Issues & Solutions

### Issue: "Cannot connect to PostgreSQL"

**Solution**:
```powershell
# Check if container is running
docker ps | Select-String postgres

# Check logs
docker logs ctu-connect-postgres

# Verify port mapping
docker port ctu-connect-postgres
```

### Issue: "Neo4j connection refused"

**Solution**:
```powershell
# Wait for Neo4j to be fully ready (takes ~30 seconds)
Start-Sleep -Seconds 30

# Check if ready
docker logs ctu-connect-neo4j | Select-String "Started"
```

### Issue: "Redis connection timeout"

**Solution**:
```powershell
# Test Redis connection
docker exec ctu-connect-redis redis-cli PING
# Should return: PONG

# Restart Redis if needed
docker restart ctu-connect-redis
```

### Issue: "Empty recommendations returned"

**Solution**:
```powershell
# Check if test data is loaded
docker exec -it ctu-connect-postgres psql -U postgres -d recommendation_db -c "SELECT COUNT(*) FROM post_embeddings;"

# Load test data if empty
.\load-test-data.ps1
```

### Issue: "Maven dependencies not downloading"

**Solution**:
```powershell
# Clear Maven cache and rebuild
mvn clean
Remove-Item -Recurse -Force ~/.m2/repository/vn/ctu/edu
mvn install -U
```

---

## Testing Workflow

### 1. Basic Flow Test

```powershell
# Get feed (should work even without Python service)
curl "http://localhost:8095/api/recommendation/feed?userId=user-001&size=5"

# Verify response has posts
```

### 2. Interaction Test

```powershell
# Like a post
curl -X POST http://localhost:8095/api/recommendation/interaction `
  -H "Content-Type: application/json" `
  -d '{"userId":"user-001","postId":"post-123","type":"LIKE"}'

# Get feed again (cache should be invalidated)
curl "http://localhost:8095/api/recommendation/feed?userId=user-001&size=5"
```

### 3. Cache Test

```powershell
# First call (cache miss)
Measure-Command { curl "http://localhost:8095/api/recommendation/feed?userId=user-002&size=10" }

# Second call (cache hit - should be faster)
Measure-Command { curl "http://localhost:8095/api/recommendation/feed?userId=user-002&size=10" }
```

---

## Next Steps

1. ✅ Basic setup working
2. 📊 Load more test data
3. 🐍 Set up Python model service (optional)
4. 📈 Monitor performance metrics
5. 🎨 Integrate with frontend
6. 🔄 Set up training pipeline

---

## API Reference

### Main Endpoints

```
GET  /api/recommendation/feed
     Query Params: userId, page (optional), size (optional)
     Returns: Personalized post recommendations

POST /api/recommendation/interaction
     Body: {userId, postId, type, viewDuration, context}
     Returns: {status, message}

POST /api/recommendation/cache/invalidate
     Query Params: userId
     Returns: {status, message}

GET  /actuator/health
     Returns: Service health status

GET  /actuator/metrics
     Returns: Performance metrics
```

### Interaction Types

- `VIEW` - User viewed the post
- `LIKE` - User liked the post
- `COMMENT` - User commented on the post
- `SHARE` - User shared the post
- `SAVE` - User saved the post
- `CLICK` - User clicked on the post
- `SKIP` - User skipped the post
- `HIDE` - User hid the post

---

## 📚 Additional Resources

- [HYBRID_ARCHITECTURE.md](./HYBRID_ARCHITECTURE.md) - Detailed architecture documentation
- [TESTING_GUIDE.md](./TESTING_GUIDE.md) - Comprehensive testing guide
- [HOW_TO_USE.md](./HOW_TO_USE.md) - Usage examples
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design overview

---

## 🎉 Success!

If you can get personalized feed recommendations, you're ready to go! The system will work with fallback ranking if Python service is not available, and will automatically use the ML model once it's set up.
