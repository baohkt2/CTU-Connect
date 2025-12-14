# 📋 KẾ HOẠCH CÀI ĐẶT KIẾN TRÚC GỢI Ý BẠN BÈ MỚI

## 🎯 Tổng quan

Tài liệu này mô tả kế hoạch cài đặt có hệ thống để thay thế **kiến trúc gợi ý kết bạn hiện tại** (rule-based trong `SocialGraphService.java`) bằng **kiến trúc Hybrid AI mới** tích hợp PhoBERT để gợi ý bạn bè thông minh hơn.

---

## 📊 Phân tích hệ thống hiện tại

### 1. Kiến trúc hiện tại (AS-IS)

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER SERVICE (Port: 8092)                    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              SocialGraphService.java                       │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │ getFriendSuggestions(userId, limit)                 │  │ │
│  │  │ ├── getMutualFriendsSuggestions() → Neo4j Query     │  │ │
│  │  │ ├── getAcademicConnectionSuggestions() → Neo4j Query│  │ │
│  │  │ ├── getFriendsOfFriendsSuggestions() → Neo4j Query  │  │ │
│  │  │ ├── getProfileViewersSuggestions() → Empty (TODO)   │  │ │
│  │  │ └── getSimilarInterestsSuggestions() → Empty (TODO) │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                    │
│                            ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     Neo4j Graph DB                         │ │
│  │   • User relationships (IS_FRIENDS_WITH)                  │ │
│  │   • Academic relationships (ENROLLED_IN, IN_BATCH)        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                    │
│                            ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     Redis Cache                            │ │
│  │   • friend_suggestions:{userId} (TTL: 6h)                 │ │
│  │   • mutual_friends:{userId1}:{userId2} (TTL: 1h)          │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Hạn chế của kiến trúc hiện tại

| Hạn chế | Mô tả | Impact |
|---------|-------|--------|
| **Không có ML** | Chỉ dựa vào rules đơn giản | Gợi ý không được cá nhân hóa sâu |
| **Missing Signals** | `getProfileViewersSuggestions()` và `getSimilarInterestsSuggestions()` trả về empty | Bỏ lỡ dữ liệu quan trọng |
| **Scoring đơn giản** | Điểm relevance chỉ dựa vào mutual friends count | Không tận dụng content similarity |
| **Không tận dụng PhoBERT** | Recommend Service có PhoBERT nhưng chỉ dùng cho Posts | Lãng phí AI capability |

### 3. Kiến trúc mới (TO-BE)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              FRIEND RECOMMENDATION SYSTEM (NEW)                              │
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          USER SERVICE (Port: 8092)                                   │   │
│  │                                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────┐ │   │
│  │  │                    SocialGraphService.java (Modified)                           │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │ getFriendSuggestions(userId, limit)                                      │  │ │   │
│  │  │  │ ├── 1. Check Redis cache                                                 │  │ │   │
│  │  │  │ ├── 2. Call RecommendService for ML suggestions (NEW)                    │  │ │   │
│  │  │  │ │      └── HTTP GET /api/recommendations/friends/{userId}                │  │ │   │
│  │  │  │ └── 3. Fallback to rule-based if service unavailable                     │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────────────────────┘  │ │   │
│  │  └────────────────────────────────────────────────────────────────────────────────┘ │   │
│  │                                     │                                               │   │
│  │                                     │ HTTP (Circuit Breaker)                        │   │
│  │                                     ▼                                               │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                                    │
│  ┌─────────────────────────────────────▼───────────────────────────────────────────────┐   │
│  │                    RECOMMENDATION SERVICE (Port: 8095)                               │   │
│  │                                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────┐ │   │
│  │  │                  FriendRecommendationController.java (NEW)                      │ │   │
│  │  │   POST /api/recommendations/friends/{userId}?limit=20                          │ │   │
│  │  └────────────────────────────┬───────────────────────────────────────────────────┘ │   │
│  │                               │                                                      │   │
│  │  ┌────────────────────────────▼───────────────────────────────────────────────────┐ │   │
│  │  │               HybridFriendRecommendationService.java (NEW)                      │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │ getFriendSuggestions(userId, limit)                                      │  │ │   │
│  │  │  │ ├── 1. Get candidate users from UserService (via UserServiceClient)      │  │ │   │
│  │  │  │ ├── 2. Get user embedding (current user)                                 │  │ │   │
│  │  │  │ ├── 3. Get candidate embeddings batch                                    │  │ │   │
│  │  │  │ ├── 4. Call Python for ML ranking                                        │  │ │   │
│  │  │  │ ├── 5. Combine ML score with graph signals                               │  │ │   │
│  │  │  │ │      ├── Content Similarity (PhoBERT): 30%                             │  │ │   │
│  │  │  │ │      ├── Mutual Friends Score: 25%                                     │  │ │   │
│  │  │  │ │      ├── Academic Score: 20%                                           │  │ │   │
│  │  │  │ │      ├── Activity Score: 15%                                           │  │ │   │
│  │  │  │ │      └── Recency Score: 10%                                            │  │ │   │
│  │  │  │ └── 6. Return ranked suggestions                                         │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────────────────────┘  │ │   │
│  │  └────────────────────────────┬───────────────────────────────────────────────────┘ │   │
│  │                               │ HTTP                                                 │   │
│  │  ┌────────────────────────────▼───────────────────────────────────────────────────┐ │   │
│  │  │                    Python AI Service (Port: 8000)                               │ │   │
│  │  │   • POST /embed/user/batch → Generate user embeddings                          │ │   │
│  │  │   • POST /similarity/users/batch → Compute user similarities                   │ │   │
│  │  │   • POST /api/friends/rank → ML-based friend ranking                           │ │   │
│  │  └────────────────────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              DATA LAYER                                              │   │
│  │                                                                                      │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────────────┐ │   │
│  │  │     Neo4j        │  │   PostgreSQL     │  │          Redis                      │ │   │
│  │  │  (user-service)  │  │ (recommend_db)   │  │   (recommend-redis:6380)           │ │   │
│  │  │                  │  │                  │  │                                    │ │   │
│  │  │ • Relationships  │  │ • user_embeddings│  │ • friend_suggestions:{userId}     │ │   │
│  │  │ • Social Graph   │  │ • user_activity  │  │   (TTL: 6 hours)                  │ │   │
│  │  │ • Mutual Friends │  │ • friend_logs    │  │ • user_embedding:{userId}         │ │   │
│  │  └──────────────────┘  └──────────────────┘  │   (TTL: 24 hours)                 │ │   │
│  │                                               └────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Kế hoạch cài đặt theo Phase

### Phase 0: Chuẩn bị môi trường (Prerequisite)
**Thời gian ước tính: 0.5 ngày**

| Task | Mô tả | Priority |
|------|-------|----------|
| 0.1 | Backup code hiện tại (`SocialGraphService.java`) | P0 |
| 0.2 | Tạo branch mới: `feature/friend-recommendation-ml` | P0 |
| 0.3 | Kiểm tra Python service đang chạy (port 8000) | P0 |
| 0.4 | Kiểm tra recommend_db PostgreSQL (port 5435) | P0 |

---

### Phase 1: Database Schema & Repository Layer
**Thời gian ước tính: 1 ngày**

#### 1.1 Tạo bảng mới trong recommend_db

**File:** `recommend-service/docker/init-db/02-friend-recommendation-schema.sql`

```sql
-- Bảng lưu user embeddings
CREATE TABLE IF NOT EXISTS user_embeddings (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id VARCHAR(36) UNIQUE NOT NULL,
    major VARCHAR(100),
    faculty VARCHAR(100),
    bio TEXT,
    interests TEXT[],
    embedding REAL[768] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uk_user_embeddings_user_id UNIQUE (user_id)
);

-- Bảng log gợi ý bạn bè
CREATE TABLE IF NOT EXISTS friend_recommendation_log (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id VARCHAR(36) NOT NULL,
    recommended_user_id VARCHAR(36) NOT NULL,
    relevance_score REAL NOT NULL,
    content_similarity REAL,
    mutual_friends_score REAL,
    academic_score REAL,
    activity_score REAL,
    suggestion_type VARCHAR(50) NOT NULL,
    suggestion_reason TEXT,
    shown_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    clicked_at TIMESTAMP,
    friend_request_sent_at TIMESTAMP,
    accepted_at TIMESTAMP,
    rejected_at TIMESTAMP
);

-- Bảng activity score của user
CREATE TABLE IF NOT EXISTS user_activity_score (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id VARCHAR(36) UNIQUE NOT NULL,
    post_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    last_activity_at TIMESTAMP,
    activity_score REAL DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_embeddings_user_id ON user_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_friend_log_user_id ON friend_recommendation_log(user_id);
CREATE INDEX IF NOT EXISTS idx_friend_log_shown_at ON friend_recommendation_log(shown_at);
CREATE INDEX IF NOT EXISTS idx_user_activity_user_id ON user_activity_score(user_id);
```

#### 1.2 Tạo Entity classes

**Files cần tạo:**

| File | Path |
|------|------|
| `UserEmbedding.java` | `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/model/entity/postgres/UserEmbedding.java` |
| `FriendRecommendationLog.java` | `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/model/entity/postgres/FriendRecommendationLog.java` |
| `UserActivityScore.java` | `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/model/entity/postgres/UserActivityScore.java` |

#### 1.3 Tạo Repository interfaces

**Files cần tạo:**

| File | Path |
|------|------|
| `UserEmbeddingRepository.java` | `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/repository/postgres/UserEmbeddingRepository.java` |
| `FriendRecommendationLogRepository.java` | `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/repository/postgres/FriendRecommendationLogRepository.java` |
| `UserActivityScoreRepository.java` | `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/repository/postgres/UserActivityScoreRepository.java` |

---

### Phase 2: Python AI Service Extensions
**Thời gian ước tính: 1 ngày**

#### 2.1 Thêm User Similarity Service

**File mới:** `recommend-service/python-model/services/user_similarity_service.py`

```python
"""
User Similarity Service for Friend Recommendations
Uses PhoBERT embeddings to compute user-to-user similarity
"""

import numpy as np
from typing import List, Dict, Tuple
from inference import get_inference_engine
import logging

logger = logging.getLogger(__name__)

class UserSimilarityService:
    def __init__(self):
        self.inference_engine = get_inference_engine()
    
    def generate_user_embedding(self, user_data: Dict) -> np.ndarray:
        """Generate embedding for a user profile"""
        return self.inference_engine.encode_user_profile(user_data)
    
    def generate_user_embeddings_batch(self, users_data: List[Dict]) -> List[np.ndarray]:
        """Generate embeddings for multiple users"""
        embeddings = []
        for user_data in users_data:
            emb = self.generate_user_embedding(user_data)
            embeddings.append(emb)
        return embeddings
    
    def compute_user_similarity(self, user1_embedding: np.ndarray, 
                                 user2_embedding: np.ndarray) -> float:
        """Compute cosine similarity between two user embeddings"""
        return self.inference_engine.compute_similarity(user1_embedding, user2_embedding)
    
    def rank_friend_candidates(self, 
                               current_user_embedding: np.ndarray,
                               candidate_embeddings: List[np.ndarray],
                               candidate_ids: List[str],
                               additional_scores: Dict[str, Dict[str, float]] = None
                              ) -> List[Dict]:
        """
        Rank friend candidates using hybrid scoring
        
        Scoring weights:
        - Content Similarity (PhoBERT): 30%
        - Mutual Friends: 25%
        - Academic Connection: 20%
        - Activity Score: 15%
        - Recency: 10%
        """
        results = []
        
        for idx, (cand_emb, cand_id) in enumerate(zip(candidate_embeddings, candidate_ids)):
            # PhoBERT similarity
            content_sim = self.compute_user_similarity(current_user_embedding, cand_emb)
            
            # Get additional scores (from Java service)
            additional = additional_scores.get(cand_id, {}) if additional_scores else {}
            mutual_score = additional.get('mutual_friends_score', 0.0)
            academic_score = additional.get('academic_score', 0.0)
            activity_score = additional.get('activity_score', 0.0)
            recency_score = additional.get('recency_score', 0.0)
            
            # Hybrid scoring
            final_score = (
                content_sim * 0.30 +
                mutual_score * 0.25 +
                academic_score * 0.20 +
                activity_score * 0.15 +
                recency_score * 0.10
            )
            
            results.append({
                'user_id': cand_id,
                'final_score': float(final_score),
                'content_similarity': float(content_sim),
                'mutual_friends_score': float(mutual_score),
                'academic_score': float(academic_score),
                'activity_score': float(activity_score),
                'recency_score': float(recency_score)
            })
        
        # Sort by final score descending
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
```

#### 2.2 Thêm endpoints mới vào server.py

**Modify:** `recommend-service/python-model/server.py`

```python
# ==================== Friend Recommendation Endpoints (NEW) ====================

class UserBatchEmbeddingRequest(BaseModel):
    users: List[UserEmbeddingRequest] = Field(..., description="List of users")

class FriendRankingRequest(BaseModel):
    current_user: UserEmbeddingRequest
    candidates: List[UserEmbeddingRequest]
    additional_scores: Optional[Dict[str, Dict[str, float]]] = None

class FriendRankingResponse(BaseModel):
    rankings: List[Dict]
    count: int

@app.post("/embed/user/batch", response_model=BatchEmbeddingResponse)
async def embed_users_batch(request: UserBatchEmbeddingRequest):
    """Generate embeddings for multiple user profiles"""
    # Implementation...

@app.post("/api/friends/rank", response_model=FriendRankingResponse)
async def rank_friend_candidates(request: FriendRankingRequest):
    """Rank friend candidates using hybrid scoring"""
    # Implementation...
```

---

### Phase 3: Java Service Layer (Recommend Service)
**Thời gian ước tính: 1.5 ngày**

#### 3.1 Tạo DTOs mới

**Files cần tạo trong `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/model/dto/`:**

| File | Mô tả |
|------|-------|
| `FriendCandidateDTO.java` | DTO cho candidate user từ User Service |
| `FriendRecommendationRequest.java` | Request object cho friend ranking |
| `FriendRecommendationResponse.java` | Response object với ranked friends |
| `UserSimilarityRequest.java` | Request cho Python similarity API |
| `UserSimilarityResponse.java` | Response từ Python similarity API |

#### 3.2 Tạo Service chính

**File mới:** `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/service/HybridFriendRecommendationService.java`

```java
@Service
@Slf4j
@RequiredArgsConstructor
public class HybridFriendRecommendationService {

    private final PythonModelServiceClient pythonModelService;
    private final UserServiceClient userServiceClient;
    private final UserEmbeddingRepository userEmbeddingRepository;
    private final UserActivityScoreRepository activityScoreRepository;
    private final FriendRecommendationLogRepository logRepository;
    private final RedisCacheService redisCacheService;

    private static final String FRIEND_SUGGESTION_CACHE = "friend_suggestions:";
    private static final int CACHE_TTL_HOURS = 6;

    /**
     * Main method: Get ML-enhanced friend suggestions
     */
    public FriendRecommendationResponse getFriendSuggestions(String userId, int limit) {
        // 1. Check cache
        // 2. Get current user profile & embedding
        // 3. Get candidate users from UserService
        // 4. Get/generate embeddings for candidates
        // 5. Calculate additional scores (mutual friends, academic, activity)
        // 6. Call Python for hybrid ranking
        // 7. Build response with suggestion reasons
        // 8. Cache results
        // 9. Log recommendations
    }

    /**
     * Calculate mutual friends score (from Neo4j via UserService)
     */
    private double calculateMutualFriendsScore(String userId, String candidateId) {
        int mutualCount = userServiceClient.getMutualFriendsCount(userId, candidateId);
        return Math.min(1.0, mutualCount / 10.0);
    }

    /**
     * Calculate academic score based on faculty/major match
     */
    private double calculateAcademicScore(UserProfile user, UserProfile candidate) {
        double score = 0.0;
        if (Objects.equals(user.getFacultyId(), candidate.getFacultyId())) score += 0.4;
        if (Objects.equals(user.getMajorId(), candidate.getMajorId())) score += 0.4;
        if (Objects.equals(user.getBatchId(), candidate.getBatchId())) score += 0.2;
        return score;
    }

    /**
     * Get activity score from database
     */
    private double getActivityScore(String userId) {
        return activityScoreRepository.findByUserId(userId)
            .map(UserActivityScore::getActivityScore)
            .orElse(0.0);
    }

    /**
     * Generate suggestion reason string
     */
    private String generateSuggestionReason(FriendSuggestionDTO suggestion) {
        List<String> reasons = new ArrayList<>();
        if (suggestion.getMutualFriendsCount() > 0) {
            reasons.add(suggestion.getMutualFriendsCount() + " bạn chung");
        }
        if (suggestion.isSameMajor()) {
            reasons.add("Cùng ngành " + suggestion.getMajorName());
        }
        if (suggestion.isSameFaculty()) {
            reasons.add("Cùng khoa " + suggestion.getFacultyName());
        }
        return String.join(" • ", reasons);
    }
}
```

#### 3.3 Tạo Controller

**File mới:** `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/controller/FriendRecommendationController.java`

```java
@RestController
@RequestMapping("/api/recommendations/friends")
@RequiredArgsConstructor
@Slf4j
public class FriendRecommendationController {

    private final HybridFriendRecommendationService friendService;

    @GetMapping("/{userId}")
    public ResponseEntity<FriendRecommendationResponse> getFriendSuggestions(
            @PathVariable String userId,
            @RequestParam(defaultValue = "20") int limit) {
        
        log.info("Getting friend suggestions for user: {}, limit: {}", userId, limit);
        FriendRecommendationResponse response = friendService.getFriendSuggestions(userId, limit);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/{userId}/feedback")
    public ResponseEntity<Void> recordFeedback(
            @PathVariable String userId,
            @RequestBody FriendFeedbackRequest feedback) {
        
        friendService.recordFeedback(userId, feedback);
        return ResponseEntity.ok().build();
    }
}
```

#### 3.4 Mở rộng UserServiceClient

**Modify:** `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/client/UserServiceClient.java`

Thêm các methods mới:
- `getFriendCandidates(String userId, int limit)` → Lấy danh sách candidate users
- `getMutualFriendsCount(String userId1, String userId2)` → Đếm mutual friends
- `getUserProfile(String userId)` → Lấy full profile

---

### Phase 4: User Service Integration
**Thời gian ước tính: 1 ngày**

#### 4.1 Thêm endpoint mới trong User Service

**Modify:** `user-service/src/main/java/com/ctuconnect/controller/FriendController.java` (hoặc tạo mới)

```java
@GetMapping("/candidates/{userId}")
public ResponseEntity<List<FriendCandidateDTO>> getFriendCandidates(
        @PathVariable String userId,
        @RequestParam(defaultValue = "100") int limit) {
    // Return users who:
    // - Are not already friends
    // - Have not blocked/been blocked
    // - Have pending friend request
    // - Same faculty/major (prioritized)
}

@GetMapping("/mutual-count/{userId1}/{userId2}")
public ResponseEntity<Integer> getMutualFriendsCount(
        @PathVariable String userId1,
        @PathVariable String userId2) {
    return ResponseEntity.ok(socialGraphService.getMutualFriendsCount(userId1, userId2));
}
```

#### 4.2 Modify SocialGraphService để call Recommend Service

**Modify:** `user-service/src/main/java/com/ctuconnect/service/SocialGraphService.java`

```java
@Service
@RequiredArgsConstructor
@Slf4j
public class SocialGraphService {

    private final UserRepository userRepository;
    private final Neo4jTemplate neo4jTemplate;
    private final RedisTemplate<String, Object> redisTemplate;
    private final RecommendServiceClient recommendServiceClient; // NEW

    private static final String FRIEND_SUGGESTIONS_CACHE = "friend_suggestions:";
    private static final int CACHE_TTL_HOURS = 6;
    
    @Value("${recommendation.friend-ml.enabled:true}")
    private boolean mlFriendSuggestionEnabled; // NEW

    /**
     * Enhanced friend suggestion - calls Recommend Service for ML-based suggestions
     */
    public List<FriendSuggestionDTO> getFriendSuggestions(String userId, int limit) {
        String cacheKey = FRIEND_SUGGESTIONS_CACHE + userId;

        // Try cache first
        List<FriendSuggestionDTO> cached = getCachedSuggestions(cacheKey);
        if (cached != null && !cached.isEmpty()) {
            return cached.stream().limit(limit).collect(Collectors.toList());
        }

        List<FriendSuggestionDTO> suggestions;

        // NEW: Try ML-based suggestions first
        if (mlFriendSuggestionEnabled) {
            try {
                suggestions = recommendServiceClient.getFriendSuggestions(userId, limit);
                if (suggestions != null && !suggestions.isEmpty()) {
                    cacheSuggestions(cacheKey, suggestions);
                    return suggestions;
                }
            } catch (Exception e) {
                log.warn("ML friend suggestion failed, falling back to rule-based: {}", e.getMessage());
            }
        }

        // Fallback to rule-based suggestions (existing code)
        suggestions = getRuleBasedSuggestions(userId, limit);
        cacheSuggestions(cacheKey, suggestions);
        return suggestions;
    }

    /**
     * Original rule-based suggestions (refactored from existing code)
     */
    private List<FriendSuggestionDTO> getRuleBasedSuggestions(String userId, int limit) {
        // ... existing code moved here ...
    }
}
```

#### 4.3 Tạo RecommendServiceClient trong User Service

**File mới:** `user-service/src/main/java/com/ctuconnect/client/RecommendServiceClient.java`

```java
@FeignClient(name = "recommend-service", fallback = RecommendServiceClientFallback.class)
public interface RecommendServiceClient {

    @GetMapping("/api/recommendations/friends/{userId}")
    List<FriendSuggestionDTO> getFriendSuggestions(
        @PathVariable("userId") String userId,
        @RequestParam("limit") int limit
    );
}
```

---

### Phase 5: Testing & Validation
**Thời gian ước tính: 1 ngày**

#### 5.1 Unit Tests

| Test Class | Coverage |
|------------|----------|
| `HybridFriendRecommendationServiceTest.java` | Service logic, scoring |
| `FriendRecommendationControllerTest.java` | API endpoints |
| `UserSimilarityServiceTest.py` | Python similarity |

#### 5.2 Integration Tests

```java
@SpringBootTest
@ActiveProfiles("test")
class FriendRecommendationIntegrationTest {

    @Test
    void testEndToEndFriendSuggestion() {
        // 1. Setup test users
        // 2. Generate embeddings
        // 3. Call API
        // 4. Verify results contain expected users
        // 5. Verify scoring is reasonable
    }

    @Test
    void testFallbackWhenPythonUnavailable() {
        // Mock Python service to be down
        // Verify fallback to rule-based works
    }
}
```

#### 5.3 Performance Tests

| Metric | Target |
|--------|--------|
| API Latency (P50) | < 200ms |
| API Latency (P99) | < 500ms |
| Cache Hit Rate | > 80% |
| Throughput | > 100 req/s |

---

### Phase 6: Deployment & Monitoring
**Thời gian ước tính: 0.5 ngày**

#### 6.1 Configuration

**application.yml (recommend-service):**
```yaml
recommendation:
  friend:
    enabled: true
    cache-ttl-hours: 6
    default-limit: 20
    scoring:
      content-weight: 0.30
      mutual-friends-weight: 0.25
      academic-weight: 0.20
      activity-weight: 0.15
      recency-weight: 0.10
```

**application.yml (user-service):**
```yaml
recommendation:
  friend-ml:
    enabled: true
    fallback-enabled: true
    timeout-ms: 2000
```

#### 6.2 Feature Flags

```yaml
# Enable/disable ML friend suggestions
feature:
  friend-recommendation:
    ml-enabled: true
    fallback-enabled: true
    ab-test-percentage: 50  # For A/B testing
```

#### 6.3 Monitoring Metrics

| Metric | Description |
|--------|-------------|
| `friend_suggestions_total` | Total API calls |
| `friend_suggestions_latency` | Response latency |
| `friend_suggestions_cache_hit` | Cache hit rate |
| `friend_suggestions_ml_success` | ML service success rate |
| `friend_suggestions_fallback` | Fallback rate |

---

## 📅 Timeline tổng hợp

```
Week 1:
├── Day 1: Phase 0 (Prep) + Phase 1 (Database)
├── Day 2: Phase 2 (Python extensions)
├── Day 3-4: Phase 3 (Java Service Layer)
└── Day 5: Phase 4 (User Service Integration)

Week 2:
├── Day 1-2: Phase 5 (Testing)
├── Day 3: Phase 6 (Deployment)
└── Day 4-5: Bug fixes & optimization
```

---

## ✅ Checklist trước khi Go-Live

- [ ] Tất cả unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks đạt target
- [ ] Feature flag hoạt động
- [ ] Fallback mechanism hoạt động
- [ ] Monitoring dashboards ready
- [ ] Documentation updated
- [ ] Rollback plan prepared

---

## 🔄 Rollback Plan

Nếu có vấn đề sau khi deploy:

1. **Quick rollback**: Disable ML via feature flag
   ```yaml
   recommendation.friend-ml.enabled: false
   ```

2. **Full rollback**: Revert to previous `SocialGraphService.java`
   ```bash
   git revert <commit-hash>
   ```

3. **Database rollback**: Không cần (new tables, không modify existing)

---

## 📚 References

- [ARCHITECTURE-CTU-CONNECT-V2.md](./ARCHITECTURE-CTU-CONNECT-V2.md) - Kiến trúc tổng quan
- [recommend-service/ARCHITECTURE.md](./recommend-service/ARCHITECTURE.md) - Recommend Service architecture
- [HybridRecommendationService.java](./recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/service/HybridRecommendationService.java) - Reference implementation

---

*Tài liệu được tạo: 14/12/2025*
*Version: 1.0*
