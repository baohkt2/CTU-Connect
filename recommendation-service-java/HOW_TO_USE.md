# 🎯 How to Use - Recommendation Service

Complete guide cách sử dụng và kiểm thử Recommendation Service từ đầu đến cuối.

---

## 📚 Documentation Structure

Đọc theo thứ tự này:

1. **HOW_TO_USE.md** ← **You are here** - Overview
2. **README_DEV.md** - Development setup
3. **QUICKSTART_TESTING.md** - Quick testing guide
4. **TESTING_GUIDE.md** - Complete testing guide

---

## 🚀 Complete Workflow

### Phase 1: Setup Environment (10 phút)

#### 1.1. Start Databases

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java
.\start-dev.ps1
```

**Verify databases are running:**
```powershell
docker-compose -f docker-compose.dev.yml ps
```

**Expected:**
- ✅ postgres-recommend-dev (port 5435)
- ✅ neo4j-recommend-dev (ports 7474, 7687)
- ✅ redis-recommend-dev (port 6379)

#### 1.2. Start Service

**Option A: IDE (Recommended)**
```
IntelliJ IDEA: Select "RecommendationService-Dev" → Run (Shift+F10)
VS Code: Press F5 → Select "RecommendationService (Dev)"
```

**Option B: Command Line**
```powershell
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

**Verify service is running:**
```bash
curl http://localhost:8095/api/recommend/health
```

**Expected response:**
```json
{"status":"UP","timestamp":"2025-12-07T10:00:00Z"}
```

---

### Phase 2: Load Test Data (5 phút)

#### 2.1. Auto Load (Easy)

```powershell
.\load-test-data.ps1
```

This will load:
- 5 users into PostgreSQL & Neo4j
- 12 posts into PostgreSQL
- Friend relationships into Neo4j
- Major/Faculty relationships into Neo4j

#### 2.2. Manual Load (If auto fails)

**PostgreSQL:**
```bash
docker exec -i postgres-recommend-dev psql -U postgres -d recommendation_db < test-data.sql
```

**Neo4j:**
1. Open: http://localhost:7474
2. Login: neo4j / password
3. Copy-paste content from `test-data.cypher`
4. Click Run

#### 2.3. Verify Data

```powershell
.\verify-test-data.ps1
```

Or manually:

**Check PostgreSQL:**
```bash
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db -c "SELECT COUNT(*) FROM post_embeddings;"
```

**Check Neo4j:**
- Open: http://localhost:7474
- Run: `MATCH (u:User) RETURN count(u);`
- Expected: 5 users

---

### Phase 3: Test APIs (15 phút)

#### 3.1. Automated Testing

```powershell
.\test-api.ps1
```

This tests:
- Health check
- Get recommendations
- Record feedback
- Actuator endpoints

#### 3.2. Manual Testing

**Test 1: Health Check**
```bash
curl http://localhost:8095/api/recommend/health
```

**Test 2: Get Recommendations (Simple)**
```bash
curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=5"
```

**Test 3: Get Recommendations (Advanced)**
```bash
curl -X POST http://localhost:8095/api/recommend/posts \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "11111111-1111-1111-1111-111111111111",
    "page": 0,
    "size": 10,
    "includeExplanation": true,
    "filters": {
      "minAcademicScore": 0.85
    }
  }'
```

**Test 4: Record Feedback**
```bash
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "11111111-1111-1111-1111-111111111111",
    "postId": "post-001",
    "feedbackType": "LIKE"
  }'
```

**Test 5: Clear Cache**
```bash
curl -X DELETE http://localhost:8095/api/recommend/cache/11111111-1111-1111-1111-111111111111
```

---

## 🎯 Understanding the Recommendation Algorithm

### Algorithm Overview

```
Final Score = 
  35% × Content Similarity +
  30% × Graph Relations +
  25% × Academic Score +
  10% × Popularity Score
```

### Components Explained

#### 1. Content Similarity (35%)
- Uses embedding vectors (PhoBERT)
- Compares post content with user interests
- Higher score = more relevant content

#### 2. Graph Relations (30%)
- **FRIEND** (weight 1.0) - Direct friends
- **SAME_MAJOR** (weight 0.8) - Same major
- **SAME_FACULTY** (weight 0.6) - Same faculty
- **SAME_BATCH** (weight 0.5) - Same batch/year

#### 3. Academic Score (25%)
- Binary classifier (0-1)
- Measures educational value
- Research papers > tutorials > social posts

#### 4. Popularity Score (10%)
- Based on engagement:
  - Likes
  - Comments
  - Shares
  - Views

### Example Calculation

**User A** wants recommendations:
- Friend with User B
- Same major as User C
- Same faculty as User D

**Post from User B:**
```
Content Similarity: 0.85
Graph Relation: 1.0 (friend)
Academic Score: 0.90
Popularity: 0.75

Final Score = 0.35×0.85 + 0.30×1.0 + 0.25×0.90 + 0.10×0.75
            = 0.2975 + 0.30 + 0.225 + 0.075
            = 0.8975 ⭐ HIGH SCORE
```

---

## 📊 Real-World Scenarios

### Scenario 1: New User (Cold Start)

**Problem:** User has no history

**Solution:**
1. Recommend popular posts
2. Recommend from same faculty/major
3. Recommend high academic content

**Test:**
```bash
curl "http://localhost:8095/api/recommend/posts?userId=new-user-id&size=10"
```

---

### Scenario 2: Active User

**User has interactions:**
- Liked 5 ML posts
- Commented on 3 scholarship posts
- Friends with 10 KTPM students

**Expected recommendations:**
- More ML content
- Scholarship opportunities
- Posts from KTPM friends

**Test:**
```bash
# Get recommendations
curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=10"

# User likes a post
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d '{...}'

# Clear cache to see updated recommendations
curl -X DELETE http://localhost:8095/api/recommend/cache/11111111-1111-1111-1111-111111111111

# Get recommendations again
curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=10"
```

---

### Scenario 3: Academic Focus

**User prefers academic content**

**Test with filter:**
```bash
curl -X POST http://localhost:8095/api/recommend/posts \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "11111111-1111-1111-1111-111111111111",
    "filters": {
      "minAcademicScore": 0.85
    }
  }'
```

**Expected:** Only research, papers, tutorials (no social posts)

---

## 🔍 Monitoring & Debugging

### 1. Check Logs

**Real-time:**
```bash
tail -f logs/recommendation-service-dev.log
```

**Filter errors:**
```bash
grep "ERROR" logs/recommendation-service-dev.log
```

**Filter specific user:**
```bash
grep "11111111-1111-1111-1111-111111111111" logs/recommendation-service-dev.log
```

### 2. Monitor Metrics

**Health:**
```bash
curl http://localhost:8095/actuator/health
```

**All metrics:**
```bash
curl http://localhost:8095/actuator/metrics
```

**Specific metric:**
```bash
curl http://localhost:8095/actuator/metrics/http.server.requests
curl http://localhost:8095/actuator/metrics/jvm.memory.used
```

### 3. Database Inspection

**PostgreSQL:**
```bash
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db

# View posts
SELECT post_id, LEFT(content, 50), academic_score FROM post_embeddings LIMIT 10;

# Top academic posts
SELECT post_id, academic_score FROM post_embeddings ORDER BY academic_score DESC LIMIT 5;
```

**Neo4j:**
- Open: http://localhost:7474

```cypher
// View user network
MATCH (u:User {userId: '11111111-1111-1111-1111-111111111111'})
OPTIONAL MATCH (u)-[r:FRIEND]->(friend)
RETURN u, r, friend;

// Count relationships
MATCH ()-[r:FRIEND]->() RETURN count(r);
```

**Redis:**
```bash
docker exec -it redis-recommend-dev redis-cli

# View all keys
KEYS *

# View user cache
GET recommend:11111111-1111-1111-1111-111111111111

# Check TTL (time to live)
TTL recommend:11111111-1111-1111-1111-111111111111
```

---

## 🎨 Using with Frontend

### Integration Example (React/Next.js)

```typescript
// services/recommendationService.ts
const API_BASE = 'http://localhost:8095/api/recommend';

export async function getRecommendations(userId: string, size: number = 10) {
  const response = await fetch(
    `${API_BASE}/posts?userId=${userId}&size=${size}`
  );
  return response.json();
}

export async function recordFeedback(
  userId: string, 
  postId: string, 
  feedbackType: 'LIKE' | 'VIEW' | 'COMMENT' | 'SHARE'
) {
  const response = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, postId, feedbackType })
  });
  return response.json();
}

// Usage in component
function RecommendedFeed({ userId }: { userId: string }) {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    getRecommendations(userId, 20).then(data => {
      setPosts(data.recommendations);
    });
  }, [userId]);

  const handleLike = (postId: string) => {
    recordFeedback(userId, postId, 'LIKE');
  };

  return (
    <div>
      {posts.map(post => (
        <PostCard 
          key={post.postId} 
          post={post} 
          onLike={() => handleLike(post.postId)}
        />
      ))}
    </div>
  );
}
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Empty Recommendations

**Причины:**
- No data in database
- User doesn't exist

**Solution:**
```powershell
.\load-test-data.ps1
```

---

### Issue 2: Slow Response

**Причины:**
- Cache not working
- Too many posts

**Solution:**
```bash
# Check cache
docker exec -it redis-recommend-dev redis-cli
> KEYS recommend:*

# Check logs for slow queries
tail -f logs/recommendation-service-dev.log | grep "SLOW"
```

---

### Issue 3: Wrong Recommendations

**Причины:**
- Missing graph relationships
- Algorithm weights not tuned

**Solution:**
```bash
# Rebuild embeddings
curl -X POST http://localhost:8095/api/recommend/embedding/rebuild

# Check Neo4j relationships
# Open: http://localhost:7474
# Run: MATCH (u:User)-[r]->(other) RETURN type(r), count(*)
```

---

## 📈 Performance Benchmarks

### Expected Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Response Time (cached) | < 50ms | 10-30ms |
| Response Time (uncached) | < 500ms | 100-300ms |
| Cache Hit Rate | > 70% | 75-85% |
| Throughput | > 100 req/s | 150-300 req/s |

### Performance Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=10"
```

---

## ✅ Production Checklist

Before deploying to production:

- [ ] Tune algorithm weights based on user feedback
- [ ] Load realistic data (1000+ posts, 100+ users)
- [ ] Test with high load (>1000 req/min)
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure proper cache TTLs
- [ ] Set up database backups
- [ ] Enable security (authentication, authorization)
- [ ] Set rate limiting
- [ ] Configure CORS properly
- [ ] Add API documentation (Swagger/OpenAPI)

---

## 🎓 Learning Path

1. **Day 1:** Setup + Basic testing (this guide)
2. **Day 2:** Understanding algorithm + Tuning weights
3. **Day 3:** Frontend integration
4. **Day 4:** Performance optimization
5. **Day 5:** Production deployment

---

## 📚 Additional Resources

### Documentation
- [TESTING_GUIDE.md](./TESTING_GUIDE.md) - Complete testing guide
- [README_DEV.md](./README_DEV.md) - Development setup
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [README.md](./README.md) - Main documentation

### Scripts
- `start-dev.ps1` - Start databases
- `stop-dev.ps1` - Stop databases
- `load-test-data.ps1` - Load test data
- `test-api.ps1` - Test APIs
- `verify-test-data.ps1` - Verify data

### Useful Links
- Neo4j Browser: http://localhost:7474
- API Health: http://localhost:8095/api/recommend/health
- Actuator: http://localhost:8095/actuator

---

## 🆘 Getting Help

**Issues?**
1. Check logs: `tail -f logs/recommendation-service-dev.log`
2. Verify databases: `docker-compose -f docker-compose.dev.yml ps`
3. Test health: `curl localhost:8095/api/recommend/health`
4. Read troubleshooting: [TESTING_GUIDE.md](./TESTING_GUIDE.md#troubleshooting)

---

## 🎉 You're Ready!

Bạn đã có đầy đủ kiến thức để sử dụng và kiểm thử Recommendation Service!

**Quick Start:**
```powershell
.\start-dev.ps1          # Start databases
# Run service in IDE
.\load-test-data.ps1     # Load test data
.\test-api.ps1           # Test APIs
```

**Happy coding! 🚀**

---

**Last Updated:** 2025-12-07  
**Version:** 1.0.0
