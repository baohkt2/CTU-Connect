# 🧪 Testing & Usage Guide - Recommendation Service

Hướng dẫn chi tiết cách sử dụng và kiểm thử Recommendation Service.

---

## 📋 Table of Contents

1. [Service Overview](#service-overview)
2. [API Endpoints](#api-endpoints)
3. [Test Data Setup](#test-data-setup)
4. [API Testing Examples](#api-testing-examples)
5. [Integration Testing](#integration-testing)
6. [Performance Testing](#performance-testing)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Service Overview

### What This Service Does

Recommendation Service cung cấp gợi ý bài viết cá nhân hóa cho người dùng dựa trên:

1. **Content Similarity** (35%) - Độ tương đồng nội dung
2. **Graph Relations** (30%) - Quan hệ xã hội (bạn bè, cùng khoa, cùng lớp)
3. **Academic Score** (25%) - Mức độ học thuật
4. **Popularity** (10%) - Lượt like, comment, share

### Architecture

```
User Request → API → Ranking Engine → [Content + Graph + Academic + Popularity]
                          ↓
              PostgreSQL + Neo4j + Redis
                          ↓
                  Sorted Recommendations
```

---

## 🔌 API Endpoints

### Base URL
```
http://localhost:8095/api/recommend
```

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/posts` | Get recommendations (simple) |
| POST | `/posts` | Get recommendations (advanced) |
| POST | `/feedback` | Record user feedback |
| POST | `/embedding/rebuild` | Rebuild embeddings (admin) |
| DELETE | `/cache/{userId}` | Clear user cache |
| GET | `/actuator/health` | Actuator health check |
| GET | `/actuator/metrics` | System metrics |

---

## 📝 Test Data Setup

### Step 1: Prepare Test Data

Tạo file `test-data.sql`:

```sql
-- Connect to PostgreSQL
-- docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db

-- Create test users (metadata only - graph in Neo4j)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    username VARCHAR(100),
    faculty_id VARCHAR(50),
    major_id VARCHAR(50),
    batch_id VARCHAR(20)
);

INSERT INTO users (id, username, faculty_id, major_id, batch_id) VALUES
('11111111-1111-1111-1111-111111111111', 'nguyen_van_a', 'CNTT', 'KTPM', '2021'),
('22222222-2222-2222-2222-222222222222', 'tran_thi_b', 'CNTT', 'KTPM', '2021'),
('33333333-3333-3333-3333-333333333333', 'le_van_c', 'CNTT', 'HTTT', '2021'),
('44444444-4444-4444-4444-444444444444', 'pham_thi_d', 'CNTT', 'KHMT', '2022');

-- Create test posts with embeddings
INSERT INTO post_embeddings (id, post_id, author_id, content, academic_score, popularity_score, created_at, updated_at)
VALUES
(gen_random_uuid(), 'post-001', '11111111-1111-1111-1111-111111111111', 
 'Nghiên cứu về thuật toán Machine Learning trong xử lý ngôn ngữ tự nhiên', 
 0.95, 0.8, NOW(), NOW()),

(gen_random_uuid(), 'post-002', '22222222-2222-2222-2222-222222222222',
 'Hướng dẫn sử dụng Spring Boot và PostgreSQL cho sinh viên CNTT',
 0.90, 0.7, NOW(), NOW()),

(gen_random_uuid(), 'post-003', '33333333-3333-3333-3333-333333333333',
 'Cuộc thi lập trình CTU Code War 2024 - Giải thưởng hấp dẫn',
 0.85, 0.9, NOW(), NOW()),

(gen_random_uuid(), 'post-004', '11111111-1111-1111-1111-111111111111',
 'Học bổng toàn phần du học Nhật Bản dành cho sinh viên CNTT',
 0.92, 0.75, NOW(), NOW()),

(gen_random_uuid(), 'post-005', '44444444-4444-4444-4444-444444444444',
 'Tips ôn thi cuối kỳ môn Cấu trúc dữ liệu và Giải thuật',
 0.88, 0.65, NOW(), NOW()),

(gen_random_uuid(), 'post-006', '22222222-2222-2222-2222-222222222222',
 'Sự kiện giao lưu sinh viên IT toàn quốc tại CTU',
 0.70, 0.95, NOW(), NOW()),

(gen_random_uuid(), 'post-007', '33333333-3333-3333-3333-333333333333',
 'Review khóa học Python cho Data Science trên Coursera',
 0.87, 0.60, NOW(), NOW()),

(gen_random_uuid(), 'post-008', '11111111-1111-1111-1111-111111111111',
 'Bài báo khoa học về Deep Learning được công bố trên IEEE',
 0.98, 0.50, NOW(), NOW());

-- Verify
SELECT post_id, LEFT(content, 50) as content_preview, academic_score, popularity_score 
FROM post_embeddings 
ORDER BY created_at DESC;
```

### Step 2: Insert Graph Data (Neo4j)

Tạo file `test-data.cypher`:

```cypher
// Clear existing data (optional)
MATCH (n) DETACH DELETE n;

// Create users
CREATE (u1:User {
  userId: '11111111-1111-1111-1111-111111111111',
  username: 'nguyen_van_a',
  facultyId: 'CNTT',
  majorId: 'KTPM',
  batchId: '2021'
})

CREATE (u2:User {
  userId: '22222222-2222-2222-2222-222222222222',
  username: 'tran_thi_b',
  facultyId: 'CNTT',
  majorId: 'KTPM',
  batchId: '2021'
})

CREATE (u3:User {
  userId: '33333333-3333-3333-3333-333333333333',
  username: 'le_van_c',
  facultyId: 'CNTT',
  majorId: 'HTTT',
  batchId: '2021'
})

CREATE (u4:User {
  userId: '44444444-4444-4444-4444-444444444444',
  username: 'pham_thi_d',
  facultyId: 'CNTT',
  majorId: 'KHMT',
  batchId: '2022'
});

// Create relationships
MATCH (u1:User {userId: '11111111-1111-1111-1111-111111111111'})
MATCH (u2:User {userId: '22222222-2222-2222-2222-222222222222'})
CREATE (u1)-[:FRIEND {since: '2021-09-01'}]->(u2);

MATCH (u1:User {userId: '11111111-1111-1111-1111-111111111111'})
MATCH (u3:User {userId: '33333333-3333-3333-3333-333333333333'})
CREATE (u1)-[:FRIEND {since: '2021-10-15'}]->(u3);

MATCH (u2:User {userId: '22222222-2222-2222-2222-222222222222'})
MATCH (u3:User {userId: '33333333-3333-3333-3333-333333333333'})
CREATE (u2)-[:FRIEND {since: '2021-09-20'}]->(u3);

// Verify
MATCH (u:User) RETURN u;
MATCH (u1:User)-[r:FRIEND]->(u2:User) RETURN u1.username, r, u2.username;
```

### Step 3: Load Test Data

```bash
# PostgreSQL
docker exec -i postgres-recommend-dev psql -U postgres -d recommendation_db < test-data.sql

# Neo4j - Copy-paste into browser at http://localhost:7474
# Or use cypher-shell:
docker exec -i neo4j-recommend-dev cypher-shell -u neo4j -p password < test-data.cypher
```

---

## 🧪 API Testing Examples

### 1. Health Check

```bash
curl http://localhost:8095/api/recommend/health
```

**Expected Response:**
```json
{
  "status": "UP",
  "timestamp": "2025-12-07T10:00:00Z",
  "version": "1.0.0"
}
```

---

### 2. Simple Recommendations (GET)

```bash
curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=5"
```

**Expected Response:**
```json
{
  "userId": "11111111-1111-1111-1111-111111111111",
  "recommendations": [
    {
      "postId": "post-001",
      "score": 0.89,
      "authorId": "11111111-1111-1111-1111-111111111111",
      "content": "Nghiên cứu về thuật toán Machine Learning...",
      "reason": "High academic content + Same author"
    },
    {
      "postId": "post-002",
      "score": 0.85,
      "authorId": "22222222-2222-2222-2222-222222222222",
      "content": "Hướng dẫn sử dụng Spring Boot...",
      "reason": "From friend + Same major"
    }
  ],
  "page": 0,
  "size": 5,
  "totalResults": 8,
  "cached": false
}
```

---

### 3. Advanced Recommendations (POST)

```bash
curl -X POST http://localhost:8095/api/recommend/posts \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "11111111-1111-1111-1111-111111111111",
    "page": 0,
    "size": 10,
    "includeExplanation": true,
    "filters": {
      "minAcademicScore": 0.8,
      "excludeAuthors": ["44444444-4444-4444-4444-444444444444"],
      "tags": ["machine-learning", "ai"]
    }
  }'
```

**Expected Response:**
```json
{
  "userId": "11111111-1111-1111-1111-111111111111",
  "recommendations": [
    {
      "postId": "post-001",
      "score": 0.89,
      "breakdown": {
        "contentSimilarity": 0.92,
        "graphRelation": 0.85,
        "academicScore": 0.95,
        "popularityScore": 0.80
      },
      "explanation": "Highly relevant academic content from same field",
      "matchedTags": ["machine-learning", "ai"]
    }
  ],
  "metadata": {
    "totalProcessed": 8,
    "totalFiltered": 6,
    "totalReturned": 5,
    "processingTimeMs": 45
  }
}
```

---

### 4. Record Feedback

```bash
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "11111111-1111-1111-1111-111111111111",
    "postId": "post-002",
    "feedbackType": "LIKE",
    "timestamp": "2025-12-07T10:30:00Z"
  }'
```

**Feedback Types:**
- `VIEW` - Xem bài viết
- `LIKE` - Thích
- `COMMENT` - Bình luận
- `SHARE` - Chia sẻ
- `HIDE` - Ẩn
- `REPORT` - Báo cáo

**Expected Response:**
```json
{
  "success": true,
  "feedbackId": "feedback-uuid-here",
  "message": "Feedback recorded successfully"
}
```

---

### 5. Clear User Cache

```bash
curl -X DELETE http://localhost:8095/api/recommend/cache/11111111-1111-1111-1111-111111111111
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Cache cleared for user: 11111111-1111-1111-1111-111111111111"
}
```

---

### 6. Rebuild Embeddings (Admin)

```bash
curl -X POST http://localhost:8095/api/recommend/embedding/rebuild
```

**Expected Response:**
```json
{
  "success": true,
  "jobId": "rebuild-job-uuid",
  "message": "Embedding rebuild started",
  "estimatedTimeMinutes": 5
}
```

---

## 🎯 Testing Scenarios

### Scenario 1: New User Gets Recommendations

**Purpose:** Test cold start problem

```bash
# Create new user without history
curl -X POST http://localhost:8095/api/recommend/posts \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "new-user-id",
    "page": 0,
    "size": 10
  }'
```

**Expected:** Popular posts + Posts from same faculty/major

---

### Scenario 2: User Interacts and Gets Better Recommendations

**Step 1: Get initial recommendations**
```bash
curl "http://localhost:8095/api/recommend/posts?userId=22222222-2222-2222-2222-222222222222&size=5"
```

**Step 2: User likes a post**
```bash
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "22222222-2222-2222-2222-222222222222",
    "postId": "post-001",
    "feedbackType": "LIKE"
  }'
```

**Step 3: Clear cache to force recalculation**
```bash
curl -X DELETE http://localhost:8095/api/recommend/cache/22222222-2222-2222-2222-222222222222
```

**Step 4: Get recommendations again**
```bash
curl "http://localhost:8095/api/recommend/posts?userId=22222222-2222-2222-2222-222222222222&size=5"
```

**Expected:** Similar posts ranked higher

---

### Scenario 3: Filter by Academic Content

```bash
curl -X POST http://localhost:8095/api/recommend/posts \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "33333333-3333-3333-3333-333333333333",
    "page": 0,
    "size": 10,
    "filters": {
      "minAcademicScore": 0.85
    }
  }'
```

**Expected:** Only academic posts (score >= 0.85)

---

### Scenario 4: Test Caching

**First request (not cached):**
```bash
time curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=10"
```

**Second request (cached):**
```bash
time curl "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=10"
```

**Expected:** Second request should be faster (~10-50ms vs 100-500ms)

---

## 🔍 Postman Collection

Tạo file `recommendation-service.postman_collection.json`:

```json
{
  "info": {
    "name": "Recommendation Service",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "http://localhost:8095/api/recommend/health",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8095",
          "path": ["api", "recommend", "health"]
        }
      }
    },
    {
      "name": "Get Recommendations (Simple)",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "http://localhost:8095/api/recommend/posts?userId={{userId}}&size=10",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8095",
          "path": ["api", "recommend", "posts"],
          "query": [
            {"key": "userId", "value": "{{userId}}"},
            {"key": "size", "value": "10"},
            {"key": "page", "value": "0"}
          ]
        }
      }
    },
    {
      "name": "Get Recommendations (Advanced)",
      "request": {
        "method": "POST",
        "header": [
          {"key": "Content-Type", "value": "application/json"}
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"userId\": \"{{userId}}\",\n  \"page\": 0,\n  \"size\": 10,\n  \"includeExplanation\": true\n}"
        },
        "url": {
          "raw": "http://localhost:8095/api/recommend/posts",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8095",
          "path": ["api", "recommend", "posts"]
        }
      }
    },
    {
      "name": "Record Feedback",
      "request": {
        "method": "POST",
        "header": [
          {"key": "Content-Type", "value": "application/json"}
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"userId\": \"{{userId}}\",\n  \"postId\": \"{{postId}}\",\n  \"feedbackType\": \"LIKE\"\n}"
        },
        "url": {
          "raw": "http://localhost:8095/api/recommend/feedback",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8095",
          "path": ["api", "recommend", "feedback"]
        }
      }
    },
    {
      "name": "Clear Cache",
      "request": {
        "method": "DELETE",
        "header": [],
        "url": {
          "raw": "http://localhost:8095/api/recommend/cache/{{userId}}",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8095",
          "path": ["api", "recommend", "cache", "{{userId}}"]
        }
      }
    }
  ],
  "variable": [
    {
      "key": "userId",
      "value": "11111111-1111-1111-1111-111111111111"
    },
    {
      "key": "postId",
      "value": "post-001"
    }
  ]
}
```

Import vào Postman: `File → Import → recommendation-service.postman_collection.json`

---

## 📊 Performance Testing

### Using Apache Bench (ab)

```bash
# Install Apache Bench
# Windows: choco install apache-httpd
# Or use from Git Bash

# Test 1: Simple load test
ab -n 1000 -c 10 "http://localhost:8095/api/recommend/posts?userId=11111111-1111-1111-1111-111111111111&size=10"

# Test 2: POST requests
ab -n 500 -c 5 -p request.json -T application/json http://localhost:8095/api/recommend/posts
```

**Create `request.json`:**
```json
{
  "userId": "11111111-1111-1111-1111-111111111111",
  "page": 0,
  "size": 10
}
```

### Expected Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Response Time (cached) | < 50ms | 10-30ms |
| Response Time (uncached) | < 500ms | 100-300ms |
| Throughput | > 100 req/s | 150-300 req/s |
| Error Rate | < 0.1% | 0% |

---

## 🐛 Troubleshooting

### Issue: Empty Recommendations

**Причины:**
1. Không có data trong database
2. User không tồn tại trong Neo4j

**Solution:**
```bash
# Check PostgreSQL
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db \
  -c "SELECT COUNT(*) FROM post_embeddings;"

# Check Neo4j
# Open http://localhost:7474
# Run: MATCH (u:User) RETURN count(u);

# If empty, load test data
docker exec -i postgres-recommend-dev psql -U postgres -d recommendation_db < test-data.sql
```

---

### Issue: Slow Response

**Причины:**
1. Cache không hoạt động
2. Database connection issues
3. Too many posts to process

**Solution:**
```bash
# Check Redis
docker exec -it redis-recommend-dev redis-cli
> KEYS recommend:*
> TTL recommend:11111111-1111-1111-1111-111111111111

# Check logs
tail -f logs/recommendation-service-dev.log | grep "SLOW"

# Check metrics
curl http://localhost:8095/actuator/metrics/http.server.requests
```

---

### Issue: Wrong Recommendations

**Причины:**
1. Algorithm weights not tuned
2. Missing graph relationships
3. Outdated embeddings

**Solution:**
```bash
# Rebuild embeddings
curl -X POST http://localhost:8095/api/recommend/embedding/rebuild

# Check graph relationships
# Neo4j: http://localhost:7474
MATCH (u:User {userId: 'your-user-id'})-[r]->(other)
RETURN type(r), count(*);

# Adjust weights in application-dev.yml
# recommendation.weights.*
```

---

## 📈 Monitoring

### Actuator Endpoints

```bash
# Health
curl http://localhost:8095/actuator/health

# Metrics
curl http://localhost:8095/actuator/metrics

# Specific metric
curl http://localhost:8095/actuator/metrics/jvm.memory.used
curl http://localhost:8095/actuator/metrics/http.server.requests

# Prometheus format
curl http://localhost:8095/actuator/prometheus
```

### Key Metrics to Monitor

1. **Response Time**: Avg, P95, P99
2. **Cache Hit Rate**: Should be > 70%
3. **Error Rate**: Should be < 0.1%
4. **Database Connections**: Monitor pool usage
5. **Memory Usage**: Heap, non-heap

---

## ✅ Testing Checklist

Before deploying:

- [ ] Health check passes
- [ ] Can get recommendations for existing user
- [ ] Can get recommendations for new user
- [ ] Feedback recording works
- [ ] Cache works (faster 2nd request)
- [ ] Filters work correctly
- [ ] Pagination works
- [ ] All error cases handled (404, 500, etc.)
- [ ] Performance meets targets
- [ ] Logs are clear and useful

---

## 🎓 Next Steps

1. **Load more realistic data** - 1000+ posts, 100+ users
2. **Tune algorithm weights** - Adjust based on user feedback
3. **Add more filters** - Date range, tags, categories
4. **Implement A/B testing** - Test different algorithms
5. **Add personalization** - Learn from user behavior

---

## 📚 Related Documentation

- **API_REFERENCE.md** - Complete API docs
- **ARCHITECTURE.md** - System architecture
- **README.md** - Main documentation
- **DEV_SETUP_GUIDE.md** - Development setup

---

**Last Updated:** 2025-12-07  
**Version:** 1.0.0
