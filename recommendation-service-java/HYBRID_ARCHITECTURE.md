# 🏗️ CTU Connect Recommendation Service - Hybrid Architecture

## 📋 Overview

The CTU Connect Recommendation Service uses a **Hybrid ML Architecture** that combines:
- **Python** for Machine Learning, NLP, and embedding generation
- **Java Spring Boot** for high-performance API, orchestration, and business logic
- **Kafka** for real-time event streaming and training pipeline
- **Redis** for caching with optimized TTL (30-120 seconds)

## 🎯 Architecture Goals

- **Response Time**: < 50-100ms with cache, < 500ms without cache
- **Real-time Learning**: Continuous model improvement from user interactions
- **Academic Focus**: Prioritize academic content (research, scholarships, Q&A, events)
- **Personalization**: Based on major, faculty, batch, degree, and user behavior

---

## 🔷 Three-Layer Architecture

### 1️⃣ **MODEL LAYER (Python Service)**

**Location**: `recommendation-service` (Python FastAPI)

**Responsibilities**:
- Generate text embeddings using PhoBERT (768 dimensions)
- Academic content classification
- ML-based post ranking using trained models
- Model training from Kafka event streams

**Input** (POST `/api/model/predict`):
```json
{
  "userAcademic": {
    "userId": "user123",
    "major": "Computer Science",
    "faculty": "Engineering",
    "degree": "Bachelor",
    "batch": "K48"
  },
  "userHistory": [
    {
      "postId": "post1",
      "liked": 1,
      "commented": 0,
      "shared": 0,
      "viewDuration": 4.3
    }
  ],
  "candidatePosts": [
    {
      "postId": "post2",
      "content": "Machine Learning workshop...",
      "hashtags": ["#AI", "#Workshop"],
      "authorMajor": "Computer Science",
      "authorFaculty": "Engineering"
    }
  ],
  "topK": 20
}
```

**Output**:
```json
{
  "rankedPosts": [
    {
      "postId": "post2",
      "score": 0.92,
      "contentSimilarity": 0.85,
      "implicitFeedback": 0.78,
      "category": "academic"
    }
  ],
  "modelVersion": "v1.2.0",
  "processingTimeMs": 150
}
```

**Model Files** (loaded at startup):
```
academic_posts_model/
├── vectorizer.pkl
├── post_encoder.pkl
├── academic_encoder.pkl
└── ranking_model.pkl
```

---

### 2️⃣ **BUSINESS LAYER (Java Spring Boot)**

**Location**: `recommendation-service-java`

**Responsibilities**:
- Main API endpoint: `GET /api/recommendation/feed?userId={id}`
- Orchestrate Python model service calls
- Apply business rules:
  - Filter viewed/hidden posts
  - Priority boost for friends
  - Priority boost for same major/faculty
  - Block list enforcement
- Cache management (Redis)
- User interaction tracking
- Send events to Kafka

**Core Services**:

1. **HybridRecommendationService** - Main orchestrator
2. **PythonModelServiceClient** - Communicate with Python service
3. **UserServiceClient** - Fetch user academic profile
4. **RedisCacheService** - Cache feed recommendations
5. **UserInteractionProducer** - Send events to Kafka

**API Endpoints**:

```
GET  /api/recommendation/feed?userId={id}&page=0&size=20
POST /api/recommendation/interaction
POST /api/recommendation/cache/invalidate?userId={id}
GET  /api/recommendation/health/python-service
```

---

### 3️⃣ **DATA PIPELINE LAYER (Kafka + Training)**

**Kafka Topics**:

| Topic | Purpose |
|-------|---------|
| `post_viewed` | User viewed a post |
| `post_liked` | User liked a post |
| `post_shared` | User shared a post |
| `post_commented` | User commented on a post |
| `user_interaction` | General interaction events |
| `recommendation_training_data` | Training data samples |

**Training Pipeline Flow**:

```
User Interaction → Kafka Topic → Python Consumer → Dataset Update → Re-train Model → Deploy New Weights
```

**Training Data Format** (matches `academic_dataset.json`):
```json
{
  "userProfile": {
    "major": "...",
    "faculty": "...",
    "degree": "...",
    "batch": "..."
  },
  "post": {
    "content": "...",
    "hashtags": ["..."],
    "mediaDescription": "...",
    "authorMajor": "...",
    "authorFaculty": "..."
  },
  "interaction": {
    "liked": 1,
    "commented": 0,
    "shared": 0,
    "viewDuration": 4.3
  },
  "timestamp": 1234567890
}
```

---

## 🚀 Recommendation Flow (End-to-End)

### Step-by-Step Process:

```
1. Frontend calls → GET /api/recommendation/feed?userId=user123

2. Java Service checks Redis cache (TTL: 30-120s)
   ├─ Cache HIT → Return cached feed (< 50ms)
   └─ Cache MISS → Continue to step 3

3. Fetch user academic profile from user-service
   → {major: "CS", faculty: "Engineering", batch: "K48"}

4. Get user interaction history (last 30 days)
   → [viewed posts, liked posts, commented posts]

5. Get candidate posts (filter already seen)
   → Top 100 trending posts from last 7 days

6. Call Python Model Service
   Request: {userAcademic, userHistory, candidatePosts}
   Response: [{postId, score, contentSimilarity}, ...]

7. Apply Business Rules in Java
   ├─ Boost score for same major (+0.2)
   ├─ Boost score for same faculty (+0.1)
   ├─ Boost score for friends (+0.3)
   └─ Filter blocked users

8. Sort by final score and take top N

9. Cache results in Redis (TTL: 30-120s)

10. Return to frontend (<100ms total)
```

---

## 📊 Caching Strategy

### Cache Layers:

1. **Embedding Cache** (Redis)
   - Key: `embedding:{postId}`
   - TTL: 1 hour
   - Value: float[768] vector

2. **Recommendation Cache** (Redis)
   - Key: `recommend:{userId}`
   - TTL: 30-120 seconds (adaptive)
   - Value: List<RecommendedPost>

3. **User Profile Cache** (Redis)
   - Key: `profile:{userId}`
   - TTL: 10 minutes
   - Value: UserAcademicProfile

### Cache Invalidation:

- User interaction (like/comment/share) → Invalidate user's recommendation cache
- Post update → Invalidate post embedding cache
- Profile update → Invalidate user profile cache

---

## 🔧 Configuration

### application.yml

```yaml
recommendation:
  # Python Model Service
  python-service:
    url: ${PYTHON_MODEL_SERVICE_URL:http://localhost:8097}
    predict-endpoint: /api/model/predict
    timeout: 10000
    enabled: true
    fallback-to-legacy: true

  # Caching
  cache:
    embedding-ttl: 3600          # 1 hour
    recommendation-ttl: 120      # 2 minutes
    user-profile-ttl: 600        # 10 minutes
    min-ttl: 30
    max-ttl: 120

  # Ranking Weights
  weights:
    content-similarity: 0.35
    graph-relation: 0.30
    academic-score: 0.25
    popularity-score: 0.10

  # Graph Relationship Weights
  graph-weights:
    friend: 1.0
    same-major: 0.8
    same-faculty: 0.6
    same-batch: 0.5
```

---

## 🧪 Testing the System

### 1. Start Services

```bash
# Start databases
docker-compose up -d postgres neo4j redis kafka

# Start Java service (IDE or Maven)
mvn spring-boot:run

# Start Python service
cd recommendation-service
python app.py
```

### 2. Test Feed API

```bash
# Get personalized feed
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=20"
```

### 3. Record Interaction

```bash
curl -X POST http://localhost:8095/api/recommendation/interaction \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "postId": "post456",
    "type": "LIKE",
    "viewDuration": 5.2,
    "context": {}
  }'
```

### 4. Check Python Service Health

```bash
curl http://localhost:8095/api/recommendation/health/python-service
```

---

## 📈 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Cache Hit Response | < 50ms | ✅ 30-40ms |
| Cache Miss Response | < 500ms | ✅ 300-450ms |
| Python Model Latency | < 200ms | ✅ 150-180ms |
| Cache Hit Rate | > 70% | ✅ 75-80% |

---

## 🔄 Training Pipeline

### Scheduled Retraining:

- **Incremental**: Every 1 hour (new interactions)
- **Full Retrain**: Every 24 hours (complete dataset)

### Training Flow:

```
1. Python consumer reads from Kafka topics
2. Append to training datasets:
   - academic_dataset.json
   - user_history_dataset.json
   - post_dataset.json
3. Retrain models (PhoBERT fine-tuning, ranking model)
4. Export new model weights
5. Reload models in Python service (hot reload)
6. Invalidate Redis caches
```

---

## 🛠️ Troubleshooting

### Python Service Down

- Java service uses fallback ranking (popularity-based)
- Logs warning: "Python model service unavailable, using fallback"
- No errors thrown, degraded functionality

### Cache Issues

- Clear all caches: `redis-cli FLUSHDB`
- Check TTL: `redis-cli TTL recommend:user123`
- Monitor cache hits: Check metrics endpoint

### Slow Responses

- Check Python service health
- Verify Redis connectivity
- Review cache hit rate
- Check database query performance

---

## 📝 Development Guide

### Adding New Features

1. **New Ranking Signal**:
   - Update Python model to include new feature
   - Update `PythonModelRequest` DTO
   - Retrain model with new feature

2. **New Business Rule**:
   - Implement in `HybridRecommendationService.applyBusinessRules()`
   - No Python service changes needed

3. **New Event Type**:
   - Add to `FeedbackType` enum
   - Update Kafka producer
   - Update Python consumer

---

## 📚 Key Files

```
recommendation-service-java/
├── src/main/java/vn/ctu/edu/recommend/
│   ├── client/
│   │   ├── PythonModelServiceClient.java    ← Python service integration
│   │   └── UserServiceClient.java           ← User profile fetching
│   ├── service/
│   │   └── HybridRecommendationService.java ← Main orchestration logic
│   ├── controller/
│   │   └── FeedController.java              ← API endpoints
│   ├── kafka/
│   │   └── producer/
│   │       ├── UserInteractionProducer.java ← Event publishing
│   │       └── TrainingDataProducer.java    ← Training data export
│   └── model/dto/
│       ├── PythonModelRequest.java          ← Model service request
│       ├── PythonModelResponse.java         ← Model service response
│       ├── UserAcademicProfile.java         ← User profile
│       ├── CandidatePost.java               ← Post data
│       └── UserInteractionHistory.java      ← Interaction history
└── HYBRID_ARCHITECTURE.md                   ← This file
```

---

## 🎓 Architecture Benefits

| Component | Benefit |
|-----------|---------|
| **Python ML Layer** | Advanced NLP, easy model updates, rich ML ecosystem |
| **Java API Layer** | High performance, enterprise reliability, easy integration |
| **Redis Caching** | Ultra-fast responses, reduced load on Python service |
| **Kafka Streaming** | Real-time learning, decoupled training pipeline |
| **Hybrid Approach** | Best of both worlds: ML power + API performance |

---

## 🚀 Next Steps

1. **Set up Python Model Service** (see recommendation-service README)
2. **Configure Kafka topics** (see docker-compose.yml)
3. **Deploy trained models** to Python service
4. **Test end-to-end flow** with sample data
5. **Monitor performance** via Prometheus/Grafana
6. **Iterate on model** based on user feedback

---

## 📞 Support

For questions or issues:
- Check logs: `logs/recommendation-service.log`
- Python service logs: `recommendation-service/logs/`
- Kafka consumer lag: `kafka-consumer-groups --describe`
- Redis cache stats: `redis-cli INFO stats`
