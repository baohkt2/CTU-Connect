# 🏗️ CTU Connect Recommendation Service - Architecture

## 1. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         API Gateway :8090                         │
│              (Routes, Load Balancing, Authentication)            │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     │ HTTP/REST
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│             Recommendation Service :8095                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Controller Layer                               │ │
│  │   /api/recommend/posts                                      │ │
│  │   /api/recommend/feedback                                   │ │
│  └───────────────────┬────────────────────────────────────────┘ │
│                      │                                            │
│  ┌───────────────────▼────────────────────────────────────────┐ │
│  │           Service Orchestration Layer                       │ │
│  │  RecommendationServiceImpl                                  │ │
│  │  ┌──────────┬──────────┬───────────┬──────────────────┐   │ │
│  │  │          │          │           │                   │   │ │
│  │  ▼          ▼          ▼           ▼                   ▼   │ │
│  │ Embedding  Graph    Ranking    Feedback           Cache  │ │
│  │ Service   Service    Engine     Service          Service │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              NLP & ML Components                            │ │
│  │  • EmbeddingService (PhoBERT)                              │ │
│  │  • AcademicClassifier (Content Classification)             │ │
│  │  • RankingEngine (Score Calculation)                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │            Event Processing (Kafka)                         │ │
│  │  • PostEventConsumer                                        │ │
│  │  • UserActionConsumer                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────┬──────────┬────────────┬─────────────┬───────────────┘
            │          │            │             │
   ┌────────▼───┐  ┌──▼─────┐  ┌──▼──────┐  ┌───▼────────┐
   │PostgreSQL  │  │ Neo4j  │  │  Redis  │  │   Kafka    │
   │+ pgvector  │  │ Graph  │  │  Cache  │  │  Events    │
   └────────────┘  └────────┘  └─────────┘  └────────────┘
        │                                          │
        │                                          │
   ┌────▼──────────┐                         ┌────▼─────────┐
   │ Post Service  │                         │PhoBERT NLP   │
   │  :8086        │                         │Service :8096 │
   └───────────────┘                         └──────────────┘
```

## 2. Component Details

### 2.1 Controller Layer
**Responsibility**: REST API endpoints, request validation, response formatting

**Components**:
- `RecommendationController`: Main REST controller
  - GET/POST `/api/recommend/posts`: Get recommendations
  - POST `/api/recommend/feedback`: Record feedback
  - POST `/api/recommend/embedding/rebuild`: Admin rebuild
  - POST `/api/recommend/rank/rebuild`: Cache rebuild

### 2.2 Service Layer
**Responsibility**: Business logic orchestration, algorithm execution

**Components**:
- `RecommendationServiceImpl`: Main service implementation
  - Orchestrates recommendation pipeline
  - Manages caching strategy
  - Handles error recovery

### 2.3 NLP Components
**Responsibility**: Text processing, embeddings, classification

**Components**:
- `EmbeddingService`: PhoBERT embedding generation
  - Calls external NLP service
  - Caches embeddings in Redis
  - Calculates cosine similarity
  
- `AcademicClassifier`: Content classification
  - ML-based classification (when available)
  - Rule-based fallback
  - Category mapping

### 2.4 Ranking Engine
**Responsibility**: Score calculation, post ranking

**Formula**:
```
final_score = α(0.35) × content_similarity +
              β(0.30) × graph_relation_score +
              γ(0.25) × academic_score +
              δ(0.10) × popularity_score
```

**Features**:
- Weighted scoring
- Diversity enforcement
- Time decay
- Personalization boost

### 2.5 Repository Layer
**Responsibility**: Data access and persistence

**PostgreSQL Repositories**:
- `PostEmbeddingRepository`: Post embeddings with pgvector
- `UserFeedbackRepository`: User feedback history
- `RecommendationCacheRepository`: Pre-computed recommendations

**Neo4j Repositories**:
- `UserGraphRepository`: Social graph queries
- `PostGraphRepository`: Content relationship queries

**Redis Cache**:
- `RedisCacheService`: Multi-level caching
  - Embedding cache (TTL: 1h)
  - Recommendation cache (TTL: 30min)
  - User profile cache (TTL: 10min)

### 2.6 Event Processing
**Responsibility**: Real-time updates via Kafka

**Consumers**:
- `PostEventConsumer`: 
  - Topics: post_created, post_updated, post_deleted
  - Actions: Generate embeddings, classify content
  
- `UserActionConsumer`:
  - Topic: user_action
  - Actions: Record feedback, update metrics, invalidate cache

## 3. Data Flow

### 3.1 Recommendation Request Flow

```
┌─────────┐
│  User   │
└────┬────┘
     │ 1. GET /api/recommend/posts?userId=xxx
     ▼
┌──────────────────┐
│   Controller     │
└────┬────────────┘
     │ 2. Validate request
     ▼
┌──────────────────┐      ┌──────────┐
│RecommendationSvc │◄────►│  Redis   │
└────┬────────────┘      └──────────┘
     │ 3. Check cache (miss)
     │
     │ 4. Get user feedback history
     ▼
┌──────────────────┐      ┌──────────┐
│   PostgreSQL     │◄─────┤Feedback  │
│                  │      │Repository│
└────┬────────────┘      └──────────┘
     │ 5. Calculate user interest vector
     │
     │ 6. Get candidate posts
     ▼
┌──────────────────┐
│  EmbeddingService│
└────┬────────────┘
     │ 7. Calculate content similarities
     │
     │ 8. Get graph relation scores
     ▼
┌──────────────────┐      ┌──────────┐
│     Neo4j        │◄─────┤  Graph   │
│                  │      │Repository│
└────┬────────────┘      └──────────┘
     │ 9. Combine scores
     ▼
┌──────────────────┐
│  RankingEngine   │
└────┬────────────┘
     │ 10. Rank & diversify
     │
     │ 11. Cache results
     ▼
┌──────────────────┐      ┌──────────┐
│      Redis       │◄─────┤  Cache   │
│                  │      │ Service  │
└────┬────────────┘      └──────────┘
     │ 12. Return response
     ▼
┌──────────────────┐
│    Response      │
└──────────────────┘
```

### 3.2 Post Creation Event Flow

```
┌──────────────┐
│ Post Service │
└──────┬───────┘
       │ 1. User creates post
       │
       │ 2. Publish post_created event
       ▼
┌──────────────┐
│    Kafka     │
└──────┬───────┘
       │ 3. Topic: post_created
       ▼
┌──────────────────┐
│PostEventConsumer │
└──────┬───────────┘
       │ 4. Consume event
       │
       │ 5. Call PhoBERT service
       ▼
┌──────────────────┐      ┌──────────┐
│  PhoBERT NLP     │◄─────┤Embedding │
│   Service        │      │ Service  │
└──────┬───────────┘      └──────────┘
       │ 6. Generate embedding vector
       │
       │ 7. Classify content
       ▼
┌──────────────────┐
│AcademicClassifier│
└──────┬───────────┘
       │ 8. Category + confidence
       │
       │ 9. Store in PostgreSQL
       ▼
┌──────────────────┐
│  PostEmbedding   │
│  + pgvector      │
└──────┬───────────┘
       │ 10. Cache in Redis
       ▼
┌──────────────────┐      ┌──────────┐
│      Redis       │◄─────┤  Cache   │
│                  │      │ Service  │
└──────┬───────────┘      └──────────┘
       │ 11. Invalidate user caches
       ▼
┌──────────────────┐
│   Complete       │
└──────────────────┘
```

### 3.3 User Feedback Flow

```
┌─────────┐
│  User   │
└────┬────┘
     │ 1. Like/Comment/Share post
     ▼
┌──────────────────┐
│   Controller     │
└────┬────────────┘
     │ 2. POST /api/recommend/feedback
     ▼
┌──────────────────┐
│RecommendationSvc │
└────┬────────────┘
     │ 3. Record feedback
     │
     │ 4. Save to PostgreSQL
     ▼
┌──────────────────┐      ┌──────────┐
│  UserFeedback    │◄─────┤Feedback  │
│    Table         │      │Repository│
└────┬────────────┘      └──────────┘
     │ 5. Update post metrics
     │
     │ 6. Calculate new popularity score
     ▼
┌──────────────────┐
│  PostEmbedding   │
│  (update counts) │
└────┬────────────┘
     │ 7. Publish user_action event
     ▼
┌──────────────────┐
│      Kafka       │
└────┬────────────┘
     │ 8. Invalidate user cache
     ▼
┌──────────────────┐      ┌──────────┐
│      Redis       │◄─────┤  Cache   │
│                  │      │ Service  │
└──────────────────┘      └──────────┘
```

## 4. Database Schema Design

### 4.1 PostgreSQL with pgvector

**post_embeddings**: Stores post content with embeddings
- Primary: Post content, embeddings, academic classification
- Indices: post_id, author_id, academic_score, vector similarity
- Optimization: IVFFlat index for fast similarity search

**user_feedback**: User interaction history
- Primary: Feedback records for reinforcement learning
- Indices: user_id + post_id, feedback_type, timestamp
- Purpose: Learn user preferences over time

**recommendation_cache**: Pre-computed recommendations
- Primary: Cached recommendation results
- Indices: user_id, updated_at
- TTL: Managed by expires_at column

### 4.2 Neo4j Graph Schema

**Nodes**:
- User: Student/faculty profiles
- Post: Content nodes
- Faculty/Major/Batch: Academic organization

**Relationships**:
- FRIEND: Social connections (weight: 1.0)
- SAME_MAJOR: Academic similarity (weight: 0.8)
- SAME_FACULTY: Department similarity (weight: 0.6)
- SAME_BATCH: Cohort similarity (weight: 0.5)
- POSTED/LIKED_BY/SHARED_BY: Content interactions

**Queries**: Optimized Cypher queries for:
- Graph relation score calculation
- Network analysis
- Collaborative filtering

## 5. Caching Strategy

### 5.1 Multi-Level Cache

```
┌──────────────────────────────────────┐
│   L1: Application Memory Cache        │
│   (Hot data, immediate access)        │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│   L2: Redis Cache                     │
│   • Embeddings (TTL: 1h)              │
│   • Recommendations (TTL: 30min)      │
│   • User Profiles (TTL: 10min)        │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│   L3: PostgreSQL Cache Table          │
│   (recommendation_cache)              │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│   Cold Storage: PostgreSQL            │
│   (All data, persistent)              │
└───────────────────────────────────────┘
```

### 5.2 Cache Invalidation Strategy

**Triggers**:
1. Post created/updated/deleted → Invalidate all recommendation caches
2. User feedback → Invalidate user's recommendation cache
3. Scheduled rebuild → Periodic cache refresh (every 5 minutes)
4. Manual trigger → Admin API endpoint

**Patterns**:
- Write-through: Update cache immediately after write
- Lazy loading: Load on cache miss
- TTL-based expiration: Automatic cleanup

## 6. Scalability & Performance

### 6.1 Horizontal Scaling

```
┌────────────────────────────────────────────┐
│        Load Balancer / API Gateway         │
└───────┬──────────┬──────────┬──────────────┘
        │          │          │
┌───────▼────┐ ┌──▼─────┐ ┌──▼─────┐
│ Recommend  │ │Recommend│ │Recommend│
│ Instance 1 │ │Instance2│ │Instance3│
└────────────┘ └─────────┘ └─────────┘
        │          │          │
        └──────────┴──────────┴───────►  Shared Redis
                                          Shared PostgreSQL
                                          Shared Neo4j
```

**Strategies**:
- Stateless service design
- Shared cache layer
- Connection pooling
- Async processing with Kafka

### 6.2 Performance Optimizations

**Database**:
- pgvector IVFFlat index for O(log n) similarity search
- Neo4j query optimization with property indices
- PostgreSQL query plan analysis

**Caching**:
- Aggressive caching of embeddings (rarely change)
- Recommendation result caching per user
- Batch database operations

**Async Processing**:
- Non-blocking Kafka consumers
- Background embedding generation
- Scheduled batch jobs

**Algorithm**:
- Early filtering of candidate posts
- Approximate similarity search
- Top-K ranking instead of full sort

## 7. Monitoring & Observability

### 7.1 Metrics (Prometheus)

**Business Metrics**:
- recommendation_requests_total
- recommendation_latency_seconds
- cache_hit_ratio
- embedding_generation_duration

**System Metrics**:
- JVM memory/CPU usage
- Database connection pool stats
- Kafka consumer lag
- Redis operations/sec

### 7.2 Logging

**Structured Logging** (JSON format):
```json
{
  "timestamp": "2025-12-07T14:30:00Z",
  "level": "INFO",
  "service": "recommendation-service",
  "trace_id": "abc123",
  "user_id": "user123",
  "action": "get_recommendations",
  "duration_ms": 125,
  "cache_hit": true
}
```

**Log Levels**:
- ERROR: Failed operations, exceptions
- WARN: Cache misses, fallback usage
- INFO: Request/response, key events
- DEBUG: Detailed algorithm steps

## 8. Security

### 8.1 Authentication & Authorization

- Integration with API Gateway authentication
- JWT token validation
- User-based access control
- Admin endpoints protection

### 8.2 Data Protection

- Sensitive data encryption at rest
- Secure communication (HTTPS/TLS)
- Database access credentials management
- Redis password protection

## 9. Failure Handling

### 9.1 Circuit Breaker Pattern

**External Services**:
- PhoBERT NLP Service: Fallback to zero vector
- Neo4j: Return 0.0 graph score
- Redis: Skip cache, direct database query

### 9.2 Graceful Degradation

**Priority Levels**:
1. Content similarity (core)
2. Popularity score (core)
3. Graph relation (nice-to-have)
4. Academic classification (nice-to-have)

**Fallback Chain**:
```
ML Classifier → Rule-based → Default category
Graph Query → Empty score → Continue
Cache → Database → Compute on-demand
```

## 10. Deployment Architecture

### 10.1 Docker Compose (Development)

```yaml
services:
  recommendation-service:
    build: ./recommendation-service-java
    ports: ["8095:8095"]
    depends_on:
      - recommend_db
      - neo4j
      - redis
      - kafka
      
  recommend_db:
    image: ankane/pgvector:latest
    ports: ["5435:5432"]
    
  neo4j:
    image: neo4j:5.13.0
    ports: ["7474:7474", "7687:7687"]
    
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### 10.2 Kubernetes (Production)

**Deployment Strategy**:
- ReplicaSet with 3+ instances
- HorizontalPodAutoscaler based on CPU/memory
- StatefulSet for databases
- Persistent volumes for data
- ConfigMap for configuration
- Secrets for credentials

**Health Checks**:
- Liveness probe: /actuator/health
- Readiness probe: /api/recommend/health
- Startup probe: Initial delay 60s

---

**Last Updated**: 2025-12-07  
**Version**: 1.0.0  
**Authors**: CTU Connect Development Team
