# CTU Connect Advanced Recommendation Service

> **AI-powered personalized post recommendation system** với PhoBERT embeddings, Neo4j graph relationships, và academic content classification.

## 🎯 Tổng Quan

Service recommendation nâng cao cho mạng xã hội CTU Connect, tập trung vào:
- **Nội dung học thuật**: Phân loại và ưu tiên nội dung học thuật
- **Cá nhân hóa**: Dựa trên ngành/khoa/lớp của sinh viên
- **AI/NLP**: Sử dụng PhoBERT cho Vietnamese text embeddings
- **Graph Ranking**: Tận dụng mối quan hệ xã hội từ Neo4j

## 🏗️ Kiến Trúc

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (8090)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│          Recommendation Service (8095)                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Controller Layer                                   │    │
│  └────────────┬───────────────────────────────────────┘    │
│               │                                              │
│  ┌────────────▼───────────────────────────────────────┐    │
│  │  Service Layer (RecommendationService)             │    │
│  │  - Orchestration                                    │    │
│  │  - Caching Logic                                    │    │
│  └────┬──────┬──────────┬────────────┬────────────────┘    │
│       │      │          │            │                      │
│  ┌────▼──┐ ┌─▼────┐ ┌──▼──────┐ ┌──▼─────────┐           │
│  │ NLP   │ │Graph │ │ Ranking │ │  Feedback  │           │
│  │Service│ │Score │ │ Engine  │ │  Learning  │           │
│  └───────┘ └──────┘ └─────────┘ └────────────┘           │
└───┬──────────┬──────────┬──────────┬──────────┬───────────┘
    │          │          │          │          │
┌───▼───┐  ┌──▼────┐  ┌──▼────┐  ┌──▼───┐  ┌──▼────┐
│PgVector│  │Neo4j  │  │Redis  │  │Kafka │  │PhoBERT│
│(Posts) │  │(Graph)│  │(Cache)│  │Events│  │ NLP   │
└────────┘  └───────┘  └───────┘  └──────┘  └───────┘
```

### Module Structure

```
recommendation-service/
├── src/main/java/vn/ctu/edu/recommend/
│   ├── config/              # Configurations
│   │   ├── RedisConfig.java
│   │   ├── WebClientConfig.java
│   │   └── KafkaConfig.java
│   ├── controller/          # REST Controllers
│   │   └── RecommendationController.java
│   ├── service/             # Business Logic
│   │   ├── RecommendationService.java
│   │   └── impl/
│   │       └── RecommendationServiceImpl.java
│   ├── repository/          # Data Access
│   │   ├── postgres/        # PostgreSQL + pgvector
│   │   ├── neo4j/           # Graph queries
│   │   └── redis/           # Cache operations
│   ├── model/               # Data Models
│   │   ├── entity/          # Database entities
│   │   ├── dto/             # API DTOs
│   │   └── enums/           # Enumerations
│   ├── nlp/                 # NLP Components
│   │   ├── EmbeddingService.java      # PhoBERT embeddings
│   │   └── AcademicClassifier.java    # Content classification
│   ├── ranking/             # Ranking Algorithm
│   │   └── RankingEngine.java
│   ├── kafka/               # Event Processing
│   │   ├── consumer/        # Event consumers
│   │   ├── producer/        # Event producers
│   │   └── event/           # Event models
│   ├── scheduler/           # Batch Jobs
│   │   └── RecommendationScheduler.java
│   ├── exception/           # Error Handling
│   │   └── GlobalExceptionHandler.java
│   └── util/                # Utilities
└── src/main/resources/
    ├── application.yml       # Main config
    └── application-docker.yml # Docker config
```

## 🧠 Core Algorithm

### Ranking Formula

```
final_score = α * content_similarity + 
              β * graph_relation_score + 
              γ * academic_score + 
              δ * popularity_score
```

Trong đó:
- **α = 0.35**: Content similarity weight (PhoBERT embeddings)
- **β = 0.30**: Graph relationship weight (Neo4j)
- **γ = 0.25**: Academic classification weight
- **δ = 0.10**: Popularity weight (engagement metrics)

### 1. Content Similarity (α = 0.35)

**PhoBERT Vietnamese Embeddings:**
- Vector 768 dimensions
- Cosine similarity giữa user interest vector và post embeddings
- Cached trong Redis với TTL 1 hour
- Stored trong PostgreSQL với pgvector extension

```java
float cosineSimilarity = embeddingService.cosineSimilarity(
    userInterestVector, 
    postEmbedding
);
```

### 2. Graph Relation Score (β = 0.30)

**Neo4j Relationship Weights:**
- FRIEND: 1.0 (bạn bè trực tiếp)
- SAME_MAJOR: 0.8 (cùng ngành)
- SAME_FACULTY: 0.6 (cùng khoa)
- SAME_BATCH: 0.5 (cùng khóa)
- FOLLOWS: 1.0 (theo dõi)

```cypher
MATCH (u:User {userId: $userId})
MATCH (p:Post {postId: $postId})
MATCH (author:User)-[:POSTED]->(p)

OPTIONAL MATCH (u)-[:FRIEND]-(author)
OPTIONAL MATCH (u)-[:MAJOR]->(m)<-[:MAJOR]-(author)
OPTIONAL MATCH (u)-[:FACULTY]->(f)<-[:FACULTY]-(author)

RETURN weighted_score
```

### 3. Academic Score (γ = 0.25)

**Classification Categories:**
- RESEARCH: Nghiên cứu khoa học
- SCHOLARSHIP: Học bổng
- QA: Hỏi đáp học thuật
- ANNOUNCEMENT: Thông báo chính thức
- EVENT: Sự kiện học thuật
- COURSE: Khóa học/môn học
- PROJECT: Dự án/đồ án
- THESIS: Luận văn/khóa luận
- NON_ACADEMIC: Nội dung không học thuật

**Methods:**
1. ML-based: PhoBERT classifier (khi có NLP service)
2. Rule-based fallback: Keyword matching

### 4. Popularity Score (δ = 0.10)

**Engagement Formula:**
```java
popularity_score = 0.4 * likes + 
                   0.3 * comments + 
                   0.2 * shares + 
                   0.1 * log(views + 1)
```

Normalized to 0-1 range với logarithmic scaling.

## 📊 Database Schema

### PostgreSQL (pgvector)

#### Table: post_embeddings
```sql
CREATE TABLE post_embeddings (
    id UUID PRIMARY KEY,
    post_id VARCHAR(255) UNIQUE NOT NULL,
    author_id VARCHAR(255) NOT NULL,
    content TEXT,
    embedding_vector vector(768),     -- pgvector extension
    academic_score FLOAT DEFAULT 0,
    academic_category VARCHAR(50),
    popularity_score FLOAT DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    faculty VARCHAR(100),
    major VARCHAR(100),
    tags TEXT[],
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    embedding_updated_at TIMESTAMP,
    
    INDEX idx_post_id (post_id),
    INDEX idx_author_id (author_id),
    INDEX idx_academic_score (academic_score)
);

-- Create pgvector index for similarity search
CREATE INDEX ON post_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops)
WITH (lists = 100);
```

#### Table: user_feedback
```sql
CREATE TABLE user_feedback (
    id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    post_id VARCHAR(255) NOT NULL,
    feedback_type VARCHAR(50) NOT NULL,
    feedback_value FLOAT NOT NULL,
    session_id VARCHAR(100),
    context JSONB,
    timestamp TIMESTAMP NOT NULL,
    
    INDEX idx_user_post (user_id, post_id),
    INDEX idx_feedback_type (feedback_type)
);
```

#### Table: recommendation_cache
```sql
CREATE TABLE recommendation_cache (
    id UUID PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    post_ids TEXT[],
    scores REAL[],
    ab_variant VARCHAR(50),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL
);
```

### Neo4j Graph Schema

#### Nodes

```cypher
// User Node
(:User {
    userId: String,
    name: String,
    faculty: String,
    major: String,
    batch: String,
    interests: [String]
})

// Post Node
(:Post {
    postId: String,
    authorId: String,
    content: String,
    category: String,
    createdAt: DateTime
})

// Faculty Node
(:Faculty {name: String})

// Major Node
(:Major {name: String})

// Batch Node
(:Batch {year: String})
```

#### Relationships

```cypher
(:User)-[:FRIEND]->(:User)           // Friendship
(:User)-[:FOLLOWS]->(:User)          // Following
(:User)-[:MAJOR]->(:Major)           // Major enrollment
(:User)-[:FACULTY]->(:Faculty)       // Faculty enrollment
(:User)-[:BATCH]->(:Batch)           // Batch/cohort
(:User)-[:POSTED]->(:Post)           // Post authorship
(:User)-[:LIKED_BY]->(:Post)         // Post likes
(:User)-[:COMMENTED_BY]->(:Post)     // Post comments
(:User)-[:SHARED_BY]->(:Post)        // Post shares
```

## 🚀 API Specification

### Base URL
```
http://localhost:8095/api/recommend
```

### Endpoints

#### 1. Get Recommendations (Simple)
```http
GET /api/recommend/posts?userId={userId}&page=0&size=20
```

**Query Parameters:**
- `userId` (required): User ID
- `page` (optional): Page number (default: 0)
- `size` (optional): Results per page (default: 20)
- `includeExplanations` (optional): Include recommendation explanations

**Response:**
```json
{
  "userId": "user123",
  "recommendations": [
    {
      "postId": "post456",
      "authorId": "author789",
      "content": "Nội dung bài viết...",
      "finalScore": 0.85,
      "contentSimilarity": 0.82,
      "graphRelationScore": 0.75,
      "academicScore": 0.90,
      "popularityScore": 0.65,
      "academicCategory": "RESEARCH",
      "rank": 1,
      "explanation": {
        "reason": "Similar to your interests, From your network, Academic content",
        "factors": ["Similar to your interests", "From your network", "Academic content"]
      }
    }
  ],
  "totalCount": 20,
  "page": 0,
  "size": 20,
  "abVariant": "computed",
  "timestamp": "2025-12-07T14:30:00",
  "processingTimeMs": 125
}
```

#### 2. Get Recommendations (Advanced)
```http
POST /api/recommend/posts
Content-Type: application/json

{
  "userId": "user123",
  "page": 0,
  "size": 20,
  "context": {
    "device": "mobile",
    "location": "campus"
  },
  "includeExplanations": true,
  "excludePostIds": ["post1", "post2"],
  "filterCategories": ["RESEARCH", "SCHOLARSHIP"]
}
```

#### 3. Record Feedback
```http
POST /api/recommend/feedback
Content-Type: application/json

{
  "userId": "user123",
  "postId": "post456",
  "feedbackType": "LIKE",
  "feedbackValue": 1.0,
  "sessionId": "session789",
  "context": {
    "source": "feed",
    "position": 3
  }
}
```

**Feedback Types:**
- `VIEW`: User viewed the post
- `CLICK`: User clicked on the post
- `LIKE`: User liked the post
- `COMMENT`: User commented
- `SHARE`: User shared the post
- `SAVE`: User saved the post
- `SKIP`: User skipped the post
- `HIDE`: User hid the post
- `REPORT`: User reported the post

#### 4. Rebuild Embeddings (Admin)
```http
POST /api/recommend/embedding/rebuild
```

#### 5. Rebuild Rankings (Admin)
```http
POST /api/recommend/rank/rebuild
```

#### 6. Invalidate User Cache
```http
DELETE /api/recommend/cache/{userId}
```

## 📡 Kafka Topics

### Consumed Topics

#### 1. post_created
```json
{
  "eventId": "evt123",
  "eventType": "POST_CREATED",
  "postId": "post456",
  "authorId": "author789",
  "content": "Nội dung bài viết...",
  "category": "RESEARCH",
  "tags": ["AI", "Machine Learning"],
  "timestamp": "2025-12-07T14:30:00"
}
```

**Actions:**
- Generate PhoBERT embedding
- Classify academic content
- Store in PostgreSQL
- Cache in Redis

#### 2. post_updated
Similar to `post_created` but updates existing embeddings.

#### 3. post_deleted
```json
{
  "eventId": "evt124",
  "eventType": "POST_DELETED",
  "postId": "post456",
  "timestamp": "2025-12-07T14:30:00"
}
```

#### 4. user_action
```json
{
  "eventId": "evt125",
  "actionType": "LIKE",
  "userId": "user123",
  "postId": "post456",
  "metadata": {"source": "feed"},
  "timestamp": "2025-12-07T14:30:00"
}
```

**Actions:**
- Record user feedback
- Update engagement metrics
- Invalidate user recommendation cache

## 🔧 Setup & Installation

### Prerequisites
- Java 17
- Maven 3.8+
- PostgreSQL 15+ with pgvector extension
- Neo4j 5.x
- Redis 7+
- Kafka 3.7+

### 1. Database Setup

#### PostgreSQL với pgvector
```bash
# Install PostgreSQL
docker run -d \
  --name recommend_db \
  -p 5435:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=recommendation_db \
  ankane/pgvector:latest

# Verify pgvector extension
psql -h localhost -p 5435 -U postgres -d recommendation_db
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Neo4j
```bash
docker run -d \
  --name neo4j-recommend \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.13.0
```

#### Redis
```bash
docker run -d \
  --name redis-recommend \
  -p 6379:6379 \
  redis:7-alpine
```

#### Kafka
```bash
# Already running in docker-compose.yml
```

### 2. Build & Run

```bash
# Clone repository
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java

# Copy environment variables
cp .env.example .env

# Build project
mvn clean package -DskipTests

# Run application
mvn spring-boot:run

# Or run with Docker
docker build -t ctu-recommend-service .
docker run -p 8095:8095 --env-file .env ctu-recommend-service
```

### 3. Verify Installation

```bash
# Check health
curl http://localhost:8095/api/recommend/health

# Check actuator
curl http://localhost:8095/actuator/health

# Check Eureka registration
curl http://localhost:8761/eureka/apps/RECOMMENDATION-SERVICE
```

## 🧪 Testing

### Unit Tests
```bash
mvn test
```

### Integration Tests
```bash
mvn verify -P integration-tests
```

### Manual API Testing
```bash
# Get recommendations
curl -X GET "http://localhost:8095/api/recommend/posts?userId=user123&size=10"

# Record feedback
curl -X POST http://localhost:8095/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "postId": "post456",
    "feedbackType": "LIKE"
  }'

# Rebuild embeddings
curl -X POST http://localhost:8095/api/recommend/embedding/rebuild
```

## 📈 Monitoring & Metrics

### Prometheus Metrics
```
http://localhost:8095/actuator/prometheus
```

**Available Metrics:**
- `recommendation_requests_total`: Total recommendation requests
- `recommendation_latency_seconds`: Request latency
- `embedding_generation_total`: Total embeddings generated
- `cache_hit_ratio`: Redis cache hit ratio
- `graph_query_duration_seconds`: Neo4j query duration

### Logs
```bash
tail -f logs/recommendation-service.log
```

## 🔄 Data Flow

### Recommendation Request Flow

```
1. User Request
   ↓
2. Check Redis Cache
   ↓ (cache miss)
3. Get User Feedback History (PostgreSQL)
   ↓
4. Calculate User Interest Vector
   ↓
5. Get Candidate Posts (PostgreSQL + pgvector)
   ↓
6. Calculate Content Similarities (PhoBERT)
   ↓
7. Calculate Graph Relation Scores (Neo4j)
   ↓
8. Rank Posts (RankingEngine)
   ↓
9. Apply Filters & Personalization
   ↓
10. Cache Results (Redis)
    ↓
11. Return Response
```

### Post Event Flow

```
Kafka: post_created event
   ↓
1. Consume Event (PostEventConsumer)
   ↓
2. Generate Embedding (PhoBERT Service)
   ↓
3. Classify Content (AcademicClassifier)
   ↓
4. Store in PostgreSQL + Neo4j
   ↓
5. Cache Embedding (Redis)
   ↓
6. Invalidate Recommendation Caches
```

## ⚙️ Configuration

### Application Properties

Key configurations trong `application.yml`:

```yaml
recommendation:
  weights:
    content-similarity: 0.35
    graph-relation: 0.30
    academic-score: 0.25
    popularity-score: 0.10
    
  graph-weights:
    friend: 1.0
    same-major: 0.8
    same-faculty: 0.6
    same-batch: 0.5
    
  cache:
    embedding-ttl: 3600        # 1 hour
    recommendation-ttl: 1800   # 30 minutes
    
  batch:
    rebuild-cron: "0 */5 * * * *"  # Every 5 minutes
```

### Environment Variables

Xem `.env.example` để biết tất cả environment variables.

## 🚢 Deployment

### Docker Compose Integration

Add to main `docker-compose.yml`:

```yaml
services:
  recommendation-service:
    build: ./recommendation-service-java
    container_name: recommendation-service
    ports:
      - "8095:8095"
    environment:
      - SPRING_PROFILES_ACTIVE=docker
      - POSTGRES_HOST=recommend_db
      - NEO4J_HOST=neo4j
      - REDIS_HOST=redis
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - EUREKA_SERVER_URL=http://eureka-server:8761/eureka/
    depends_on:
      - eureka-server
      - recommend_db
      - neo4j
      - redis
      - kafka
    networks:
      - ctuconnect-network

  recommend_db:
    image: ankane/pgvector:latest
    container_name: recommend_db
    ports:
      - "5435:5432"
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=recommendation_db
    volumes:
      - recommend_db_data:/var/lib/postgresql/data
    networks:
      - ctuconnect-network

volumes:
  recommend_db_data:
```

## 📚 Additional Documentation

- [Algorithm Details](./docs/ALGORITHM.md)
- [Database Schema](./docs/DATABASE.md)
- [API Examples](./docs/API_EXAMPLES.md)
- [Performance Tuning](./docs/PERFORMANCE.md)
- [Troubleshooting](./docs/TROUBLESHOOTING.md)

## 🔮 Future Enhancements

1. **Deep Learning Models:**
   - Fine-tune PhoBERT specifically for CTU academic content
   - Implement collaborative filtering with neural networks
   - Add multi-task learning for engagement prediction

2. **Advanced Features:**
   - Real-time personalization with online learning
   - Multi-armed bandit for A/B testing
   - Contextual recommendations (time, location, device)
   - Diversity optimization algorithms

3. **Scalability:**
   - Horizontal scaling with Kubernetes
   - Distributed embedding computation
   - Sharded PostgreSQL for large-scale data

4. **Analytics:**
   - Recommendation quality metrics dashboard
   - A/B test framework
   - User engagement analytics

## 👥 Authors

CTU Connect Development Team

## 📄 License

Internal project for Can Tho University

---

**Note**: This is a production-ready implementation integrated with existing CTU Connect infrastructure. Ensure all dependencies are running before starting the service.
