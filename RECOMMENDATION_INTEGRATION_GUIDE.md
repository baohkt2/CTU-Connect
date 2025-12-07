# 🔌 HƯỚNG DẪN TÍCH HỢP RECOMMENDATION SERVICE

## 📋 Tổng quan

Hướng dẫn cách tích hợp Recommendation Service với các services khác trong hệ thống CTU Connect.

---

## 🏗️ Kiến trúc tích hợp

```
┌─────────────┐
│   Client    │
│  (React)    │
└──────┬──────┘
       │
       ↓
┌─────────────────┐
│  API Gateway    │ Port 8090
│  (Port 8090)    │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ↓         ↓          ↓          ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│  Auth  │ │  User  │ │  Post  │ │Recommendation│
│Service │ │Service │ │Service │ │   Service    │
│  8081  │ │  8082  │ │  8083  │ │     8095     │
└────────┘ └────┬───┘ └───┬────┘ └──────┬───────┘
               │          │              │
               └──────────┴──────────────┘
                          │
                     ┌────┴─────┐
                     ↓          ↓
              ┌──────────┐ ┌─────────┐
              │  Neo4j   │ │ MongoDB │
              │  (Users) │ │ (Posts) │
              └──────────┘ └─────────┘
```

---

## 1️⃣ Tích hợp với API Gateway

### Bước 1: Thêm route trong API Gateway

**File:** `api-gateway/src/main/resources/application.yml`

```yaml
spring:
  cloud:
    gateway:
      routes:
        # Recommendation Service Routes
        - id: recommendation-service
          uri: lb://recommendation-service
          predicates:
            - Path=/api/recommendation/**
          filters:
            - StripPrefix=0
            - name: CircuitBreaker
              args:
                name: recommendationCircuitBreaker
                fallbackUri: forward:/fallback/recommendation
```

### Bước 2: Đăng ký với Eureka

**File:** `recommendation-service-java/src/main/resources/application.yml`

```yaml
eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/
    register-with-eureka: true
    fetch-registry: true
  instance:
    instance-id: ${spring.application.name}:${random.value}
    prefer-ip-address: true
```

### Bước 3: Test qua API Gateway

```powershell
# Thay vì gọi trực tiếp
curl http://localhost:8095/api/recommendation/feed?userId=user123

# Gọi qua API Gateway
curl http://localhost:8090/api/recommendation/feed?userId=user123
```

---

## 2️⃣ Tích hợp với User Service

Recommendation Service cần thông tin user từ User Service để đưa ra gợi ý phù hợp.

### Cách 1: Lấy từ Neo4j trực tiếp (Hiện tại)

**File:** `UserProfileService.java`

```java
@Service
public class UserProfileService {
    
    private final Neo4jTemplate neo4jTemplate;
    
    public UserProfile getUserProfile(String userId) {
        String cypher = """
            MATCH (u:User {userId: $userId})
            OPTIONAL MATCH (u)-[:HAS_MAJOR]->(m:Major)
            OPTIONAL MATCH (u)-[:HAS_INTEREST]->(i:Interest)
            OPTIONAL MATCH (u)-[:FRIEND_WITH]->(f:User)
            RETURN u, m, collect(DISTINCT i) as interests, 
                   collect(DISTINCT f) as friends
        """;
        
        return neo4jTemplate.findOne(cypher, 
            Map.of("userId", userId), 
            UserProfile.class)
            .orElse(null);
    }
}
```

### Cách 2: Gọi User Service qua REST (Tương lai)

**File:** `UserServiceClient.java`

```java
@Service
public class UserServiceClient {
    
    @Value("${services.user-service.url}")
    private String userServiceUrl;
    
    private final RestTemplate restTemplate;
    
    public UserProfile getUserProfile(String userId) {
        String url = userServiceUrl + "/api/users/" + userId + "/profile";
        
        try {
            return restTemplate.getForObject(url, UserProfile.class);
        } catch (Exception e) {
            log.error("Failed to get user profile from User Service", e);
            // Fallback to Neo4j
            return getUserProfileFromNeo4j(userId);
        }
    }
}
```

**Configuration:**

```yaml
services:
  user-service:
    url: http://localhost:8082  # hoặc lb://user-service
```

---

## 3️⃣ Tích hợp với Post Service

Recommendation Service cần lấy danh sách posts để đưa ra gợi ý.

### Cách 1: Lấy từ MongoDB trực tiếp (Hiện tại)

**File:** `CandidatePostService.java`

```java
@Service
public class CandidatePostService {
    
    private final MongoTemplate mongoTemplate;
    
    public List<Post> getCandidatePosts(UserProfile userProfile, int limit) {
        Query query = new Query();
        
        // Filter by user's major
        if (userProfile.getMajor() != null) {
            query.addCriteria(
                Criteria.where("targetMajor").in(userProfile.getMajor())
            );
        }
        
        // Filter by recent posts (last 30 days)
        query.addCriteria(
            Criteria.where("createdAt")
                .gte(LocalDateTime.now().minusDays(30))
        );
        
        // Exclude user's own posts
        query.addCriteria(
            Criteria.where("authorId").ne(userProfile.getUserId())
        );
        
        query.limit(limit);
        query.with(Sort.by(Sort.Direction.DESC, "createdAt"));
        
        return mongoTemplate.find(query, Post.class);
    }
}
```

### Cách 2: Gọi Post Service qua REST (Tương lai)

**File:** `PostServiceClient.java`

```java
@Service
public class PostServiceClient {
    
    @Value("${services.post-service.url}")
    private String postServiceUrl;
    
    private final RestTemplate restTemplate;
    
    public List<Post> getCandidatePosts(String userId, int limit) {
        String url = postServiceUrl + "/api/posts/candidates" +
                    "?userId=" + userId + "&limit=" + limit;
        
        try {
            PostResponse response = restTemplate.getForObject(url, PostResponse.class);
            return response.getPosts();
        } catch (Exception e) {
            log.error("Failed to get posts from Post Service", e);
            return Collections.emptyList();
        }
    }
}
```

---

## 4️⃣ Kafka Event Integration

Recommendation Service lắng nghe các events để cập nhật recommendations real-time.

### Events được lắng nghe:

#### 1. User Interaction Events

**Topic:** `user.interaction`

**Payload:**
```json
{
  "eventType": "POST_LIKED",
  "userId": "user123",
  "postId": "post456",
  "timestamp": "2024-12-07T12:00:00Z"
}
```

**Consumer:**

```java
@Service
@Slf4j
public class UserInteractionConsumer {
    
    private final UserInteractionService interactionService;
    private final CacheService cacheService;
    
    @KafkaListener(
        topics = "user.interaction",
        groupId = "recommendation-group"
    )
    public void handleUserInteraction(UserInteractionEvent event) {
        log.info("Received user interaction: {}", event);
        
        // Update user interaction history
        interactionService.recordInteraction(
            event.getUserId(),
            event.getPostId(),
            event.getEventType()
        );
        
        // Invalidate user's recommendation cache
        cacheService.invalidateUserCache(event.getUserId());
    }
}
```

#### 2. Post Created Events

**Topic:** `post.created`

**Payload:**
```json
{
  "postId": "post789",
  "authorId": "user123",
  "content": "Nghiên cứu về AI...",
  "category": "research",
  "timestamp": "2024-12-07T12:00:00Z"
}
```

**Consumer:**

```java
@KafkaListener(
    topics = "post.created",
    groupId = "recommendation-group"
)
public void handlePostCreated(PostCreatedEvent event) {
    log.info("Received new post: {}", event.getPostId());
    
    // Generate embedding for new post
    embeddingService.generateEmbedding(event.getPostId(), event.getContent());
    
    // Invalidate related users' caches
    cacheService.invalidateRelatedUsersCache(event.getAuthorId());
}
```

#### 3. User Profile Updated Events

**Topic:** `user.profile.updated`

**Payload:**
```json
{
  "userId": "user123",
  "updatedFields": ["major", "interests"],
  "timestamp": "2024-12-07T12:00:00Z"
}
```

**Consumer:**

```java
@KafkaListener(
    topics = "user.profile.updated",
    groupId = "recommendation-group"
)
public void handleProfileUpdated(ProfileUpdatedEvent event) {
    log.info("User profile updated: {}", event.getUserId());
    
    // Refresh user profile in cache
    userProfileService.refreshUserProfile(event.getUserId());
    
    // Invalidate user's recommendations
    cacheService.invalidateUserCache(event.getUserId());
}
```

---

## 5️⃣ Client Frontend Integration

### React Component Example

**File:** `RecommendedFeed.jsx`

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const RecommendedFeed = ({ userId }) => {
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchRecommendations();
    }, [userId]);

    const fetchRecommendations = async () => {
        try {
            setLoading(true);
            const response = await axios.get(
                `/api/recommendation/feed`,
                {
                    params: {
                        userId: userId,
                        size: 20
                    },
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                }
            );
            
            setRecommendations(response.data.recommendations);
            setError(null);
        } catch (err) {
            console.error('Failed to fetch recommendations:', err);
            setError('Không thể tải gợi ý bài viết');
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <LoadingSpinner />;
    if (error) return <ErrorMessage message={error} />;

    return (
        <div className="recommended-feed">
            <h2>Bài viết gợi ý cho bạn</h2>
            {recommendations.map(rec => (
                <PostCard 
                    key={rec.postId} 
                    post={rec}
                    score={rec.score}
                    reason={rec.reason}
                />
            ))}
        </div>
    );
};

export default RecommendedFeed;
```

### API Service

**File:** `services/recommendationService.js`

```javascript
import api from './api';

export const recommendationService = {
    // Get personalized feed
    getFeed: async (userId, size = 20) => {
        const response = await api.get('/recommendation/feed', {
            params: { userId, size }
        });
        return response.data;
    },

    // Get similar posts
    getSimilarPosts: async (postId, size = 5) => {
        const response = await api.get(`/recommendation/similar/${postId}`, {
            params: { size }
        });
        return response.data;
    },

    // Get trending posts
    getTrending: async (category, size = 10) => {
        const response = await api.get('/recommendation/trending', {
            params: { category, size }
        });
        return response.data;
    },

    // Track user interaction
    trackInteraction: async (userId, postId, interactionType) => {
        await api.post('/recommendation/interaction', {
            userId,
            postId,
            interactionType
        });
    }
};
```

---

## 6️⃣ Authentication & Authorization

### Thêm Security vào Recommendation Service

**File:** `SecurityConfig.java`

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/actuator/**").permitAll()
                .requestMatchers("/api/recommendation/**").authenticated()
                .anyRequest().authenticated()
            )
            .oauth2ResourceServer(oauth2 -> oauth2.jwt());
        
        return http.build();
    }
}
```

### Extract User from JWT

```java
@Component
public class JwtUtil {
    
    public String extractUserIdFromToken(String token) {
        // Parse JWT and extract userId claim
        Claims claims = Jwts.parser()
            .setSigningKey(secretKey)
            .parseClaimsJws(token)
            .getBody();
        
        return claims.get("userId", String.class);
    }
}
```

### Controller với Authentication

```java
@RestController
@RequestMapping("/api/recommendation")
public class RecommendationController {
    
    @GetMapping("/feed")
    public ResponseEntity<RecommendationResponse> getFeed(
            @AuthenticationPrincipal Jwt jwt,
            @RequestParam(defaultValue = "20") int size) {
        
        String userId = jwt.getClaim("userId");
        
        RecommendationResponse response = recommendationService.getFeed(userId, size);
        return ResponseEntity.ok(response);
    }
}
```

---

## 7️⃣ Caching Strategy

### Multi-level Caching

```java
@Service
public class CacheService {
    
    private final RedisTemplate<String, Object> redisTemplate;
    
    // Level 1: User feed cache (2 minutes)
    public void cacheUserFeed(String userId, List<Recommendation> recommendations) {
        String key = "recommendation:feed:" + userId;
        redisTemplate.opsForValue().set(key, recommendations, 2, TimeUnit.MINUTES);
    }
    
    // Level 2: Post embedding cache (1 hour)
    public void cachePostEmbedding(String postId, float[] embedding) {
        String key = "recommendation:embedding:" + postId;
        redisTemplate.opsForValue().set(key, embedding, 1, TimeUnit.HOURS);
    }
    
    // Level 3: User profile cache (10 minutes)
    public void cacheUserProfile(String userId, UserProfile profile) {
        String key = "recommendation:profile:" + userId;
        redisTemplate.opsForValue().set(key, profile, 10, TimeUnit.MINUTES);
    }
    
    // Invalidate user cache
    public void invalidateUserCache(String userId) {
        redisTemplate.delete("recommendation:feed:" + userId);
        redisTemplate.delete("recommendation:profile:" + userId);
    }
}
```

---

## 8️⃣ API Endpoints Summary

### Available Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/recommendation/feed` | Get personalized feed | ✅ |
| GET | `/api/recommendation/similar/{postId}` | Get similar posts | ✅ |
| GET | `/api/recommendation/trending` | Get trending posts | ❌ |
| POST | `/api/recommendation/interaction` | Track user interaction | ✅ |
| GET | `/api/recommendation/stats` | Get recommendation stats | ✅ |

### Request/Response Examples

#### 1. Get Feed

**Request:**
```http
GET /api/recommendation/feed?userId=user123&size=20
Authorization: Bearer <token>
```

**Response:**
```json
{
  "userId": "user123",
  "recommendations": [
    {
      "postId": "post456",
      "score": 0.87,
      "title": "Nghiên cứu về AI",
      "content": "...",
      "reason": "Based on your interest in AI",
      "author": {
        "userId": "user789",
        "name": "Nguyen Van A"
      }
    }
  ],
  "totalCount": 20,
  "modelUsed": "hybrid-v1",
  "timestamp": "2024-12-07T12:00:00Z"
}
```

#### 2. Get Similar Posts

**Request:**
```http
GET /api/recommendation/similar/post123?size=5
Authorization: Bearer <token>
```

**Response:**
```json
{
  "sourcePostId": "post123",
  "similarPosts": [
    {
      "postId": "post456",
      "similarity": 0.92,
      "title": "...",
      "content": "..."
    }
  ]
}
```

---

## 9️⃣ Testing Integration

### Integration Test Example

```java
@SpringBootTest
@AutoConfigureMockMvc
class RecommendationIntegrationTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @Test
    void testGetFeedIntegration() throws Exception {
        // Given
        String userId = "test_user";
        String token = generateTestToken(userId);
        
        // When & Then
        mockMvc.perform(get("/api/recommendation/feed")
                .param("userId", userId)
                .param("size", "10")
                .header("Authorization", "Bearer " + token))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.userId").value(userId))
                .andExpect(jsonPath("$.recommendations").isArray())
                .andExpect(jsonPath("$.recommendations.length()").value(10));
    }
}
```

---

## 🔟 Deployment Checklist

Trước khi deploy vào production:

- [ ] Đã test tất cả endpoints
- [ ] Đã config Eureka đúng
- [ ] Đã setup authentication/authorization
- [ ] Đã config Kafka topics
- [ ] Đã setup monitoring (Prometheus, Grafana)
- [ ] Đã config logging
- [ ] Đã setup database indexes
- [ ] Đã config cache TTL phù hợp
- [ ] Đã test performance (load testing)
- [ ] Đã setup circuit breaker
- [ ] Đã config rate limiting
- [ ] Đã test failover scenarios

---

## 📚 Tài liệu liên quan

- `RECOMMENDATION_DEV_SETUP_VN.md` - Hướng dẫn setup development
- `RECOMMENDATION_QUICK_START.md` - Quick start guide
- `RECOMMENDATION_ARCHITECTURE_EXPLAINED.md` - Chi tiết kiến trúc
- `test-recommendation-dev.ps1` - Script test tự động

---

**🎉 Chúc bạn tích hợp thành công!**
