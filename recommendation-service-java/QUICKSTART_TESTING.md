# ⚡ Quick Start - Testing Guide

Hướng dẫn nhanh test Recommendation Service trong 5 phút.

---

## 🚀 Quick Test (3 Bước)

### Bước 1: Load Test Data (1 phút)

```powershell
# Load dữ liệu mẫu vào databases
.\load-test-data.ps1
```

**Dữ liệu được tạo:**
- 5 users (Nguyễn Văn A, Trần Thị B, Lê Văn C, Phạm Thị D, Hoàng Văn E)
- 12 posts (từ academic đến social)
- Quan hệ bạn bè trong Neo4j
- Cùng khoa/ngành/lớp

### Bước 2: Test APIs (2 phút)

```powershell
# Chạy test script tự động
.\test-api.ps1
```

**Hoặc test thủ công:**

```bash
# 1. Health check
curl http://localhost:8095/api/recommend/health

# 2. Get recommendations
curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=5"

# 3. Record feedback
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d "{\"userId\":\"11111111-1111-1111-1111-111111111111\",\"postId\":\"post-001\",\"feedbackType\":\"LIKE\"}"
```

### Bước 3: Verify Results

```powershell
# Kiểm tra dữ liệu
.\verify-test-data.ps1
```

---

## 📊 Sample Response

### Get Recommendations Response

```json
{
  "userId": "11111111-1111-1111-1111-111111111111",
  "recommendations": [
    {
      "postId": "post-001",
      "score": 0.89,
      "content": "Nghiên cứu về thuật toán Machine Learning...",
      "authorId": "11111111-1111-1111-1111-111111111111",
      "academicScore": 0.95,
      "popularityScore": 0.8
    },
    {
      "postId": "post-002",
      "score": 0.85,
      "content": "Hướng dẫn sử dụng Spring Boot...",
      "authorId": "22222222-2222-2222-2222-222222222222",
      "academicScore": 0.90,
      "popularityScore": 0.7
    }
  ],
  "page": 0,
  "size": 5,
  "totalResults": 12,
  "cached": false
}
```

---

## 🧪 Test Scenarios

### Scenario 1: User Gets Personalized Recommendations

**User:** Nguyễn Văn A (KTPM K2021)

```bash
curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=10"
```

**Expected:**
- Posts từ bạn bè (Trần Thị B, Lê Văn C)
- Posts cùng ngành KTPM
- High academic score posts

---

### Scenario 2: Filter by Academic Score

```bash
curl -X POST http://localhost:8095/api/recommend/posts \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "11111111-1111-1111-1111-111111111111",
    "page": 0,
    "size": 10,
    "filters": {
      "minAcademicScore": 0.85
    }
  }'
```

**Expected:** Only posts with academic_score >= 0.85

---

### Scenario 3: Test Caching

**First request (slow):**
```bash
time curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=5"
```

**Second request (fast, from cache):**
```bash
time curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=5"
```

**Expected:** Second request ~10x faster

---

## 🎯 Test Data Overview

### Users

| User ID | Username | Faculty | Major | Batch |
|---------|----------|---------|-------|-------|
| 11111...1111 | nguyen_van_a | CNTT | KTPM | 2021 |
| 22222...2222 | tran_thi_b | CNTT | KTPM | 2021 |
| 33333...3333 | le_van_c | CNTT | HTTT | 2021 |
| 44444...4444 | pham_thi_d | CNTT | KHMT | 2022 |
| 55555...5555 | hoang_van_e | CNTT | KTPM | 2021 |

### Posts

| Post ID | Content | Academic | Popularity |
|---------|---------|----------|------------|
| post-001 | ML research | 0.95 | 0.80 |
| post-002 | Spring Boot guide | 0.90 | 0.70 |
| post-003 | Code competition | 0.85 | 0.90 |
| post-004 | Scholarship Japan | 0.92 | 0.75 |
| post-005 | Exam tips | 0.88 | 0.65 |
| post-006 | IT event | 0.70 | 0.95 |
| post-007 | Python course review | 0.87 | 0.60 |
| post-008 | IEEE paper | 0.98 | 0.50 |

### Relationships

```
nguyen_van_a (KTPM) --FRIEND--> tran_thi_b (KTPM)
nguyen_van_a (KTPM) --FRIEND--> le_van_c (HTTT)
tran_thi_b (KTPM) --FRIEND--> le_van_c (HTTT)
nguyen_van_a (KTPM) --FRIEND--> hoang_van_e (KTPM)
```

---

## 🔍 Useful Queries

### Check PostgreSQL Data

```sql
-- Connect
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db

-- Count posts
SELECT COUNT(*) FROM post_embeddings;

-- View posts
SELECT post_id, LEFT(content, 40), academic_score, popularity_score 
FROM post_embeddings 
ORDER BY academic_score DESC;

-- Top academic posts
SELECT post_id, academic_score FROM post_embeddings 
WHERE academic_score > 0.9 
ORDER BY academic_score DESC;
```

### Check Neo4j Data

Open: http://localhost:7474

```cypher
// Count users
MATCH (u:User) RETURN count(u);

// View all users
MATCH (u:User) RETURN u;

// View friendships
MATCH (u1:User)-[r:FRIEND]->(u2:User) 
RETURN u1.username, u2.username;

// User A's network
MATCH (u:User {userId: '11111111-1111-1111-1111-111111111111'})
OPTIONAL MATCH (u)-[:FRIEND]->(friend)
OPTIONAL MATCH (u)-[:STUDIES_MAJOR]->(m)<-[:STUDIES_MAJOR]-(sameMajor)
RETURN u.username, collect(DISTINCT friend.username) as friends, 
       collect(DISTINCT sameMajor.username) as same_major;
```

### Check Redis Cache

```bash
# Connect
docker exec -it redis-recommend-dev redis-cli

# View all keys
KEYS *

# View specific cache
GET recommend:11111111-1111-1111-1111-111111111111

# Check TTL
TTL recommend:11111111-1111-1111-1111-111111111111

# Clear cache
DEL recommend:11111111-1111-1111-1111-111111111111
```

---

## 🐛 Troubleshooting

### "Empty recommendations"

```bash
# Check if data exists
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db -c "SELECT COUNT(*) FROM post_embeddings;"

# Reload data
.\load-test-data.ps1
```

### "Service not responding"

```bash
# Check service status
curl http://localhost:8095/actuator/health

# Check logs
tail -f logs/recommendation-service-dev.log
```

### "Database connection error"

```bash
# Check containers
docker-compose -f docker-compose.dev.yml ps

# Restart databases
docker-compose -f docker-compose.dev.yml restart
```

---

## 📚 Next Steps

1. **Read full guide:** [TESTING_GUIDE.md](./TESTING_GUIDE.md)
2. **API reference:** Check all available endpoints
3. **Advanced testing:** Performance, load testing
4. **Integrate with frontend:** Use in CTU Connect app

---

## 💡 Pro Tips

1. **Use Postman:** Import `recommendation-service.postman_collection.json`
2. **Monitor logs:** `tail -f logs/recommendation-service-dev.log`
3. **Check metrics:** `curl localhost:8095/actuator/metrics`
4. **Neo4j Browser:** Visualize graph at `http://localhost:7474`
5. **Clear cache:** Test fresh results with `DELETE /cache/{userId}`

---

## ✅ Quick Checklist

Test these to ensure everything works:

- [ ] Health check returns UP
- [ ] Can get recommendations for user A
- [ ] Can record feedback
- [ ] Cache works (2nd request faster)
- [ ] Can filter by academic score
- [ ] Actuator endpoints work

---

**Need help?** Check [TESTING_GUIDE.md](./TESTING_GUIDE.md) for detailed docs.

---

**Last Updated:** 2025-12-07
