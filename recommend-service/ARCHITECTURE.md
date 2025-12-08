# 🎯 KIẾN TRÚC HỆ THỐNG RECOMMENDATION - CTU CONNECT

## I. MỤC TIÊU HỆ THỐNG RECOMMENDATION

Hệ thống gợi ý bài viết tập trung vào tính **học thuật**, **cá nhân hóa**, và **hành vi người dùng**, bao gồm:

* Gợi ý bài viết mới cho News Feed
* Gợi ý bài viết phù hợp với chuyên ngành, khoa, môn học
* Gợi ý học liệu (notes, tài liệu, bài post học thuật)
* Gợi ý bạn bè theo học tập
* Gợi ý cộng đồng học thuật

---

## II. KIẾN TRÚC TỔNG QUAN

Hệ thống được chia thành **3 tầng chính**:

```
┌─────────────────────────┐
│      Application Layer   │  (Java Spring Boot – Post-Service, User-Service)
└─────────────┬───────────┘
              │
┌─────────────▼──────────────┐
│    Recommendation-Service   │  (Python/Java hybrid)
│  - API Gateway for recommend
│  - Feature Processing
│  - Ranking Pipeline
└─────────────┬──────────────┘
              │
┌─────────────▼────────────────┐
│        AI Model Layer         │
│  - PhoBERT Semantic Encoder   │
│  - Content Embedding          │
│  - User Embedding             │
│  - Candidate Scoring          │
└─────────────┬────────────────┘
              │
┌─────────────▼────────────────┐
│       Data Layer              │
│  - PostgreSQL (posts)         │
│  - Neo4j (users & relations)  │
│  - Redis (cache embedding)    │
│  - Kafka (real-time events)   │
└──────────────────────────────┘
```

---

## III. MÔ HÌNH AI TRONG HỆ THỐNG

Hệ thống sử dụng **PhoBERT (transformer-based)** để xử lý ngôn ngữ tiếng Việt.

### Model Components:
* `pytorch_model.bin` - Pre-trained PhoBERT weights
* `config.json` - Model configuration
* `tokenizer_config.json` - Tokenizer settings
* `vocab.txt` - Vietnamese vocabulary

### Model Usage:
* Tạo embedding cho bài viết
* Tạo embedding cho mô tả học thuật của user
* So khớp bài viết và user bằng cosine similarity

---

## IV. KẾT HỢP PYTHON + JAVA TRONG KIẾN TRÚC

### ⚙ PYTHON SERVICE (Port 8000)

**Vai trò:** AI Inference Engine

**Chức năng:**
* Phục vụ AI Inference
* Chạy model PhoBERT
* Sinh embedding vectors
* Nhận batch dữ liệu từ Java
* Tính cosine similarity

**Endpoints:**
* `POST /embed/post` - Generate embedding for single post
* `POST /embed/post/batch` - Generate embeddings for multiple posts
* `POST /embed/user` - Generate embedding for user profile
* `POST /similarity` - Compute cosine similarity between two embeddings
* `POST /similarity/batch` - Batch similarity computation
* `GET /health` - Health check

### ☕ JAVA SERVICE (Port 8081)

**Vai trò:** Recommendation Orchestrator

**Chức năng:**
* Xử lý REST API cho frontend
* Orchestrate pipeline recommendation
* Gọi Python service qua HTTP
* Dùng embedding để rank kết quả
* Quản lý cache (Redis)
* Xử lý Kafka events
* Kết nối PostgreSQL & Neo4j

**Endpoints:**
* `GET /api/recommendations/feed` - Get personalized feed
* `GET /api/recommendations/academic` - Get academic recommendations
* `GET /api/recommendations/users` - Get user recommendations
* `POST /api/recommendations/refresh` - Refresh user embeddings

---

## V. CẤU TRÚC THỨ MỤC

```
recommend-service/
│
├── java-api/                           # Java Spring Boot Service
│   ├── src/main/java/com/ctuconnect/recommend/
│   │   ├── controller/                 # REST Controllers
│   │   │   ├── RecommendationController.java
│   │   │   └── EmbeddingController.java
│   │   ├── service/                    # Business Logic
│   │   │   ├── RecommendationService.java
│   │   │   ├── RankingService.java
│   │   │   ├── CandidateService.java
│   │   │   └── CacheService.java
│   │   ├── client/                     # External Service Clients
│   │   │   ├── PythonInferenceClient.java
│   │   │   ├── PostServiceClient.java
│   │   │   └── UserServiceClient.java
│   │   ├── dto/                        # Data Transfer Objects
│   │   │   ├── PostEmbeddingDTO.java
│   │   │   ├── UserEmbeddingDTO.java
│   │   │   └── RecommendationDTO.java
│   │   ├── config/                     # Configurations
│   │   │   ├── RedisConfig.java
│   │   │   ├── KafkaConfig.java
│   │   │   └── RestTemplateConfig.java
│   │   ├── consumer/                   # Kafka Consumers
│   │   │   ├── PostEventConsumer.java
│   │   │   └── UserEventConsumer.java
│   │   └── model/                      # Domain Models
│   │       ├── PostEmbedding.java
│   │       └── UserEmbedding.java
│   └── pom.xml
│
├── python-model/                       # Python Inference Service
│   ├── model/                          # PhoBERT Model Files
│   │   └── academic_posts_model/
│   │       ├── pytorch_model.bin
│   │       ├── config.json
│   │       └── tokenizer/
│   ├── inference.py                    # Core Inference Engine
│   ├── server.py                       # FastAPI Server
│   ├── requirements.txt                # Python Dependencies
│   └── config.py                       # Configuration
│
└── docker/                             # Docker Configurations
    ├── docker-compose.yml              # Service Orchestration
    ├── recommend-java.Dockerfile       # Java Service Image
    └── recommend-python.Dockerfile     # Python Service Image
```

---

## VI. LUỒNG HOẠT ĐỘNG CHI TIẾT

### 1. LUỒNG TẠO EMBEDDING CHO BÀI VIẾT

**Khi user tạo bài viết mới:**

#### Step 1: Post-Service → Kafka
```json
{
  "postId": "123",
  "userId": "u1",
  "content": "Mạng máy tính chương 4 - giao thức TCP...",
  "title": "Giao thức TCP/IP"
}
```

#### Step 2: Recommend-Service Consumer
* Nhận event từ Kafka topic `post-created`
* Parse post data
* Gửi request sang Python service

#### Step 3: Python Inference
```python
# Tokenize content
tokens = tokenizer(text, max_length=256, padding=True, truncation=True)

# Run PhoBERT model
outputs = model(**tokens)

# Extract [CLS] token embedding
embedding = outputs.last_hidden_state[:, 0, :]
```

#### Step 4: Java xử lý
* Nhận embedding từ Python
* Lưu vào PostgreSQL (persistent storage)
* Cache vào Redis (fast access)
* Index vào search engine (optional)

---

### 2. LUỒNG TẠO EMBEDDING CHO USER

**User profile có các thông tin:**
* major (chuyên ngành)
* faculty (khoa)
* courses (danh sách môn học)
* skills (kỹ năng)
* bio (giới thiệu bản thân)
* interaction history (lịch sử tương tác)

#### Process:
1. **Java gom dữ liệu:** Tổng hợp tất cả thông tin user từ Neo4j
2. **Gửi sang Python:** POST /embed/user
3. **Python xử lý:** Tạo text representation và generate embedding
4. **Java lưu trữ:** Cache embedding vào Redis với TTL

---

### 3. LUỒNG GỢI Ý NEWS FEED

**Khi user mở app và request feed:**

#### Step 1: Lấy User Embedding
```java
// Try get from Redis cache
UserEmbedding userEmb = redisTemplate.opsForValue().get("user:emb:" + userId);

if (userEmb == null) {
    // Generate new embedding via Python service
    userEmb = pythonClient.generateUserEmbedding(userId);
    // Cache for 1 hour
    redisTemplate.opsForValue().set("user:emb:" + userId, userEmb, 1, TimeUnit.HOURS);
}
```

#### Step 2: Lấy Candidates (Bài viết ứng viên)
```java
// Get candidates from multiple sources
List<Post> candidates = new ArrayList<>();

// 1. Posts from same major
candidates.addAll(postRepository.findByMajor(user.getMajor(), limit));

// 2. Posts from friends
candidates.addAll(postRepository.findByUserIds(user.getFriendIds(), limit));

// 3. Trending academic posts
candidates.addAll(postRepository.findTrendingAcademic(limit));

// 4. Recent posts from same faculty
candidates.addAll(postRepository.findByFaculty(user.getFaculty(), limit));
```

#### Step 3: Tính điểm Similarity
```java
// Get post embeddings from Redis/DB
List<PostEmbedding> postEmbeddings = getPostEmbeddings(candidates);

// Compute similarities via Python service
SimilarityScores scores = pythonClient.computeBatchSimilarity(
    userEmb, 
    postEmbeddings
);
```

#### Step 4: Ranking với Multiple Factors
```java
for (Post post : candidates) {
    double score = 0.0;
    
    // 1. Semantic similarity (50%)
    score += 0.5 * cosineSimilarity(userEmb, postEmb);
    
    // 2. Time decay (20%)
    score += 0.2 * timeDecayScore(post.getCreatedAt());
    
    // 3. Academic relevance (20%)
    score += 0.2 * academicRelevanceScore(post, user);
    
    // 4. Social signals (10%)
    score += 0.1 * socialScore(post.getLikes(), post.getComments());
    
    post.setRecommendationScore(score);
}

// Sort by score descending
candidates.sort((a, b) -> Double.compare(b.getScore(), a.getScore()));
```

#### Step 5: Trả về Top N
```java
return candidates.stream()
    .limit(20)
    .collect(Collectors.toList());
```

---

## VII. THÀNH PHẦN CORE QUAN TRỌNG

### A. Python Inference Engine

**File:** `python-model/inference.py`

**Class:** `PhoBERTInference`

**Methods:**
* `encode_text(text)` - Encode single text
* `encode_batch(texts)` - Batch encoding
* `encode_post(content, title)` - Post-specific encoding
* `encode_user_profile(user_data)` - User-specific encoding
* `compute_similarity(emb1, emb2)` - Cosine similarity
* `compute_batch_similarity(query, candidates)` - Batch similarity

### B. FastAPI Server

**File:** `python-model/server.py`

**Endpoints:**
* `POST /embed/post` - Single post embedding
* `POST /embed/post/batch` - Batch post embedding
* `POST /embed/user` - User profile embedding
* `POST /similarity` - Pairwise similarity
* `POST /similarity/batch` - Batch similarity
* `GET /health` - Health check

### C. Java Recommendation Engine

**Key Services:**

1. **RecommendationService**
   * Main orchestrator
   * Handles recommendation logic
   * Combines multiple signals

2. **RankingService**
   * Implements ranking algorithms
   * Time decay calculation
   * Multi-factor scoring

3. **CandidateService**
   * Fetches candidate posts
   * Applies initial filters
   * Manages candidate pool

4. **CacheService**
   * Redis operations
   * Embedding cache management
   * Result cache

5. **PythonInferenceClient**
   * HTTP client for Python service
   * Handles embedding requests
   * Error handling & retry logic

---

## VIII. DATA FLOW DIAGRAM

```
┌──────────┐
│  User    │
└────┬─────┘
     │ Request Feed
     ▼
┌────────────────┐
│  Java Service  │
│  (Port 8081)   │
└────┬───────────┘
     │
     ├─► Redis (Check user embedding cache)
     │
     ├─► Neo4j (Get user profile & relations)
     │
     ├─► PostgreSQL (Get candidate posts)
     │
     ├─► Python Service (Generate embeddings)
     │   └─► PhoBERT Model
     │
     ├─► Compute Rankings
     │
     └─► Return Recommendations
```

---

## IX. DEPLOYMENT

### Local Development

```bash
# Start Python service
cd python-model
pip install -r requirements.txt
uvicorn server:app --reload --port 8000

# Start Java service
cd java-api
./mvnw spring-boot:run
```

### Docker Deployment

```bash
cd docker
docker-compose up -d
```

### Services:
* Python Inference: `http://localhost:8000`
* Java API: `http://localhost:8081`

---

## X. PERFORMANCE CONSIDERATIONS

### Caching Strategy
* User embeddings: 1 hour TTL
* Post embeddings: Permanent (updated on edit)
* Recommendation results: 5 minutes TTL

### Batch Processing
* Process posts in batches of 32
* Batch similarity computation
* Async embedding generation

### Scalability
* Python service: Horizontal scaling with load balancer
* Java service: Multiple instances with Eureka
* Redis cluster for distributed cache
* Kafka for event streaming

---

## XI. MONITORING & METRICS

### Key Metrics
* Embedding generation time
* Recommendation latency
* Cache hit rate
* Model inference throughput
* API response time

### Logging
* Request/response logs
* Error tracking
* Performance metrics
* User interactions

---

## XII. FUTURE ENHANCEMENTS

1. **Advanced Models**
   * Fine-tune PhoBERT for academic content
   * Multi-modal embeddings (text + images)
   * Graph neural networks for social connections

2. **Personalization**
   * Real-time user interest tracking
   * Session-based recommendations
   * A/B testing framework

3. **Content Understanding**
   * Topic modeling
   * Academic classification
   * Quality scoring

4. **Social Signals**
   * Collaborative filtering
   * Friend influence
   * Community recommendations
