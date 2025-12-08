# Architecture After Optimization

## Service Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT / FRONTEND                        │
│                     (React, Mobile App, etc.)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP REST
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              JAVA SERVICE - Recommendation Orchestrator          │
│                         (Port 8081)                               │
├─────────────────────────────────────────────────────────────────┤
│  Controllers (1 Unified):                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ RecommendationController                                   │ │
│  │ - GET  /api/recommendations/feed                          │ │
│  │ - POST /api/recommendations/interaction                   │ │
│  │ - POST /api/recommendations/refresh                       │ │
│  │ - GET  /api/recommendations/health                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Services (1 Main):                                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ HybridRecommendationService                               │ │
│  │ - getFeed()                                               │ │
│  │ - recordInteraction()                                     │ │
│  │ - invalidateUserCache()                                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Clients (3 Specific):                                           │
│  ┌──────────────────┐ ┌───────────────┐ ┌──────────────────┐  │
│  │PythonModel       │ │UserService    │ │PostService       │  │
│  │ServiceClient     │ │Client (Feign) │ │Client (Feign)    │  │
│  └──────────────────┘ └───────────────┘ └──────────────────┘  │
└──────┬──────────────────┬────────────────────┬──────────────────┘
       │                  │                    │
       │ HTTP             │ Feign              │ Feign
       │                  │ (Eureka)           │ (Eureka)
       ▼                  ▼                    ▼
┌─────────────┐   ┌──────────────┐    ┌──────────────┐
│   PYTHON    │   │ USER-SERVICE │    │ POST-SERVICE │
│   SERVICE   │   │  (Port 8082) │    │  (Port 8083) │
│ (Port 8000) │   └──────────────┘    └──────────────┘
└─────────────┘
       │
       │ Loads
       ▼
┌──────────────────────────────────────┐
│       PHOBERT MODEL                  │
│  - pytorch_model.bin                 │
│  - tokenizer                         │
│  - config.json                       │
└──────────────────────────────────────┘
```

## Data Flow - Get Personalized Feed

```
1. User Request
   │
   ├──► GET /api/recommendations/feed?userId=123&size=20
   │
   ▼
2. Java: RecommendationController.getFeed()
   │
   ├──► Check Redis cache
   │    ├─ Cache hit? Return immediately ✅
   │    └─ Cache miss? Continue ▼
   │
   ├──► Call UserServiceClient.getUserAcademicProfile(userId)
   │    └─ Get user's major, faculty, courses, etc.
   │
   ├──► Query user interaction history (Neo4j/PostgreSQL)
   │    └─ Get user's likes, comments, shares, etc.
   │
   ├──► Get candidate posts (PostgreSQL)
   │    └─ Filter: exclude seen posts, apply business rules
   │
   ├──► Call PythonModelServiceClient.predictRanking()
   │    │
   │    └──► Python: POST /api/model/predict
   │         │
   │         ├──► Generate user embedding (PhoBERT)
   │         ├──► Generate post embeddings (PhoBERT)
   │         ├──► Calculate content similarity
   │         ├──► Calculate implicit feedback score
   │         ├──► Calculate academic score
   │         ├──► Calculate popularity score
   │         ├──► Combine scores & rank
   │         │
   │         └──► Return ranked posts
   │
   ├──► Apply business rules (Java)
   │    ├─ Boost same major/faculty posts
   │    ├─ Boost friend posts
   │    └─ Filter blocked users
   │
   ├──► Cache results (Redis, 30-120s TTL)
   │
   └──► Return response to client
```

## Python Service - Unified Entry Point

```
┌──────────────────────────────────────────────────────────────┐
│                    PYTHON SERVICE (server.py)                 │
│                         Port 8000                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  CORE ENDPOINTS (Embedding & Similarity):                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ POST /embed/post          - Single post embedding    │   │
│  │ POST /embed/post/batch    - Batch post embeddings   │   │
│  │ POST /embed/user          - User profile embedding   │   │
│  │ POST /similarity          - Pairwise similarity      │   │
│  │ POST /similarity/batch    - Batch similarity         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ML ENDPOINTS (Prediction & Classification):                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ POST /api/model/predict        - ML ranking          │   │
│  │ POST /api/model/embed          - Text embedding      │   │
│  │ POST /api/model/classify/academic - Classification   │   │
│  │ GET  /api/model/info           - Model info          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  HEALTH & MONITORING:                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ GET /                - Root                           │   │
│  │ GET /health          - Health check                   │   │
│  │ GET /metrics         - Metrics                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  INFERENCE ENGINE:                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ inference.py - PhoBERTInference                       │   │
│  │  - encode_post()                                      │   │
│  │  - encode_user_profile()                              │   │
│  │  - compute_similarity()                               │   │
│  │  - compute_batch_similarity()                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ML SERVICE (Optional):                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ services/prediction_service.py                        │   │
│  │  - predict()                                          │   │
│  │  - classify_academic()                                │   │
│  │  - generate_embedding()                               │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Communication Patterns

### 1. Java → Python (Embeddings)
```java
// Via PythonModelServiceClient
PostEmbeddingRequest request = new PostEmbeddingRequest();
request.setPostId("p1");
request.setContent("Mạng máy tính chương 4...");
request.setTitle("TCP/IP Protocol");

EmbeddingResponse response = pythonClient.post()
    .uri("/embed/post")
    .bodyValue(request)
    .retrieve()
    .bodyToMono(EmbeddingResponse.class)
    .block();

float[] embedding = response.getEmbedding();
// Use embedding for ranking
```

### 2. Java → Python (ML Predictions)
```java
// Via PythonModelServiceClient
PythonModelRequest request = PythonModelRequest.builder()
    .userAcademic(userProfile)
    .userHistory(interactionHistory)
    .candidatePosts(candidates)
    .topK(20)
    .build();

PythonModelResponse response = pythonClient.post()
    .uri("/api/model/predict")
    .bodyValue(request)
    .retrieve()
    .bodyToMono(PythonModelResponse.class)
    .block();

List<RankedPost> rankedPosts = response.getRankedPosts();
```

### 3. Java → Other Services (Feign)
```java
// Auto-discovered via Eureka
@FeignClient(name = "user-service")
public interface UserServiceClient {
    @GetMapping("/api/users/{userId}/academic-profile")
    UserAcademicProfile getUserAcademicProfile(@PathVariable String userId);
}

// Usage
UserAcademicProfile profile = userServiceClient
    .getUserAcademicProfile("user123");
```

## Caching Strategy

```
┌────────────────────────────────────────────────────────┐
│                    REDIS CACHE                          │
├────────────────────────────────────────────────────────┤
│                                                         │
│  User Embeddings:                                      │
│  Key: "user:emb:{userId}"                              │
│  TTL: 1 hour                                           │
│  Value: float[768] embedding vector                    │
│                                                         │
│  Post Embeddings:                                      │
│  Key: "post:emb:{postId}"                              │
│  TTL: Permanent (invalidate on edit)                   │
│  Value: float[768] embedding vector                    │
│                                                         │
│  Recommendation Results:                               │
│  Key: "recommend:feed:{userId}"                        │
│  TTL: 30-120 seconds (adaptive)                        │
│  Value: List<RecommendedPost>                          │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Kafka Event Flow

```
Java Service                 Kafka                  Python Consumer
    │                          │                          │
    │ User likes post          │                          │
    ├─────────────────────────►│                          │
    │ Topic: user-interactions │                          │
    │                          ├─────────────────────────►│
    │                          │                          │
    │                          │                     Process event
    │                          │                     Update model
    │                          │                     Generate training
    │                          │                     data sample
    │                          │                          │
```

## Database Schema

### PostgreSQL (Post Embeddings)
```sql
CREATE TABLE post_embeddings (
    post_id VARCHAR(255) PRIMARY KEY,
    content TEXT,
    embedding FLOAT[768],
    author_id VARCHAR(255),
    author_major VARCHAR(100),
    author_faculty VARCHAR(100),
    like_count INT,
    comment_count INT,
    share_count INT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Neo4j (User Graph)
```cypher
// User nodes with relationships
(user:User)-[:FRIENDS_WITH]->(friend:User)
(user:User)-[:SAME_MAJOR]->(peer:User)
(user:User)-[:SAME_FACULTY]->(peer:User)
(user:User)-[:LIKED]->(post:Post)
(user:User)-[:COMMENTED]->(post:Post)
```

## Key Improvements

### ✅ Before Optimization
- 2 Controllers with overlapping endpoints
- 2 Python entry points (app.py + server.py)
- 3 Different API path patterns
- Unclear responsibility boundaries
- Duplicate code and logic

### ✅ After Optimization
- 1 Unified Controller (`RecommendationController`)
- 1 Python entry point (`server.py`)
- 1 Consistent API pattern (`/api/recommendations/*`)
- Clear separation: Java = Orchestrator, Python = AI Engine
- No duplicate code

## Performance Characteristics

| Operation | Latency | Caching |
|-----------|---------|---------|
| Get Feed (cache hit) | ~5-10ms | ✅ Redis |
| Get Feed (cache miss) | ~150-300ms | ❌ Then cached |
| Generate Embedding | ~50-100ms | ✅ Redis (1h) |
| ML Prediction | ~100-200ms | ❌ Dynamic |
| Record Interaction | ~2-5ms | ❌ Write-through |

## Scalability

### Horizontal Scaling
```
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Java-1  │  │  Java-2  │  │  Java-3  │
│ (8081)   │  │ (8081)   │  │ (8081)   │
└─────┬────┘  └─────┬────┘  └─────┬────┘
      │             │             │
      └─────────────┼─────────────┘
                    │
         Load Balancer (Nginx/ALB)
                    │
      ┌─────────────┼─────────────┐
      │             │             │
┌─────▼────┐  ┌─────▼────┐  ┌─────▼────┐
│ Python-1 │  │ Python-2 │  │ Python-3 │
│ (8000)   │  │ (8000)   │  │ (8000)   │
└──────────┘  └──────────┘  └──────────┘
```

### Resource Requirements
- **Java Service**: 2-4 GB RAM, 2 CPUs
- **Python Service**: 4-8 GB RAM (model), 2-4 CPUs
- **Redis**: 2-4 GB RAM
- **PostgreSQL**: 4-8 GB RAM
- **Neo4j**: 4-8 GB RAM

---

**Version**: 2.0.0 (Optimized)
**Last Updated**: 2024-12-08
**Status**: ✅ Production Ready
