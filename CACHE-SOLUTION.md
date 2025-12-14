# Giải Pháp Cache cho Recommend Service

## ⚠️ ROLLBACK NOTICE (2024-12-14)

**Cơ chế "Viewed Items Tracking" đã bị vô hiệu hóa theo yêu cầu.**

Lý do: Cache nên trả về cùng kết quả cho đến khi hết hạn hoặc được invalidate thủ công. Không nên tự động ẩn các items đã xem.

### Các thay đổi còn được giữ lại:
1. ✅ Fix Redis connection (localhost thay vì redis hostname)
2. ✅ Fix 422 error Python friend ranking (camelCase/snake_case schema)
3. ✅ Filter bài viết của bản thân user
4. ✅ Refresh cache endpoint (invalidate only, không clear viewed items)

### Các thay đổi đã rollback:
1. ❌ Viewed items tracking (markFriendsAsViewed, markPostsAsViewed)
2. ❌ Filter viewed items from cache
3. ❌ Auto-invalidate cache when all items viewed
4. ❌ refreshSuggestions/refreshUserFeed methods (replaced with simple invalidate)

---

## Vấn Đề Gốc

1. **422 Error từ Python**: Java gửi `currentUser` (camelCase) nhưng Python expect `current_user` (snake_case)
2. **Redis connection issue**: user-service không kết nối được Redis do config mặc định sai
3. **User thấy bài viết của chính mình**: Không filter authorId
4. **Missing refresh endpoint**: Frontend gọi API refresh nhưng chưa cài đặt đầy đủ ở server

## Giải Pháp Triển Khai

### 1. Fix Redis Connection (user-service)

**File**: `user-service/src/main/java/com/ctuconnect/config/RedisConfig.java`

```java
// Thay đổi: localhost thay vì redis (Docker hostname)
@Value("${spring.data.redis.host:localhost}")  // Trước: redis
private String redisHost;
```

### 2. Fix Python Schema cho Friend Ranking

**File**: `recommend-service/python-model/server.py`

Vấn đề: Java gửi `currentUser`, `additionalScores`, `topK` (camelCase) nhưng Python expect snake_case.

Giải pháp: Schema accept cả 2 conventions:

```python
class UserProfileData(BaseModel):
    userId: Optional[str] = Field(None)  # camelCase from Java
    user_id: Optional[str] = Field(None)  # snake_case
    
    @property
    def effective_user_id(self) -> str:
        return self.userId or self.user_id or "unknown"

class FriendRankingRequest(BaseModel):
    currentUser: Optional[UserProfileData] = Field(None, alias="current_user")
    current_user: Optional[UserProfileData] = Field(None)
    
    @property
    def effective_current_user(self):
        return self.currentUser or self.current_user
```

### 3. Bỏ qua bài viết của bản thân

**File**: `recommend-service/java-api/.../HybridRecommendationService.java`

```java
return posts.stream()
    .filter(post -> !excludePostIds.contains(post.getPostId()))
    // NEW: Exclude user's own posts
    .filter(post -> !userId.equals(post.getAuthorId()))
    .limit(limit)
    ...
```

```
Request → Get viewed friends từ Redis
        → Check cache (filter out viewed)
        → If filtered cache empty → Fetch fresh từ ML/DB
        → Mark returned items as viewed
        → Return to client
```

### 4. Post Recommendations Cache Flow

**File**: `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/service/HybridRecommendationService.java`

```
Request → Get viewed posts từ Redis
        → Check cache (filter out viewed)
        → If filtered cache empty → Fetch fresh từ ML
        → Mark returned items as viewed
        → Return to client
```

### 5. Refresh Endpoints

**Friend Suggestions**:
- `POST /api/recommendations/friends/{userId}/refresh`
- Clears cache AND viewed items
- User sees all suggestions again

**Post Recommendations**:
- `POST /api/recommendations/refresh?userId={userId}`
- Clears cache AND viewed posts
- User sees all posts again

**User Service Proxy**:
- `POST /api/users/friend-suggestions/refresh`
- Calls recommend-service refresh endpoint

## Cấu Hình TTL

| Loại Cache | TTL | Mô tả |
|------------|-----|-------|
| Friend Suggestions Cache | 6 hours | Gợi ý kết bạn |
| Post Recommendations Cache | 30-120 seconds | Feed bài viết (dynamic) |
| Viewed Friends | 24 hours | Items đã hiển thị |
| Viewed Posts | 24 hours | Posts đã hiển thị |

## Diagram

```
┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│   Client    │────▶│  user-service │────▶│ recommend-  
│  (Frontend) │     │               │     │   service   │
└─────────────┘     └───────────────┘     └──────┬──────┘
                                                  │
                    ┌─────────────────────────────┴─────┐
                    │              Redis                │
                    │  ┌─────────────────────────────┐  │
                    │  │ recommend:userId (cache)    │  │
                    │  │ friend_suggestions:userId   │  │
                    │  │ viewed:friends:userId (SET) │  │
                    │  │ viewed:posts:userId (SET)   │  │
                    │  └─────────────────────────────┘  │
                    └───────────────────────────────────┘
```

## Test Commands

```bash
# Test Redis connection
docker exec redis redis-cli PING

# Check viewed items
docker exec redis redis-cli SMEMBERS "viewed:friends:userId"
docker exec redis redis-cli SMEMBERS "viewed:posts:userId"

# Clear viewed items manually
docker exec redis redis-cli DEL "viewed:friends:userId"
docker exec redis redis-cli DEL "viewed:posts:userId"

# Full refresh (for recommend-redis)
docker exec ctu-recommend-redis redis-cli -a recommend_redis_pass FLUSHALL
```

## Restart Services

```powershell
# 1. Restart Python model service (QUAN TRỌNG - schema đã thay đổi)
cd d:\LVTN\CTU-Connect-demo\recommend-service\python-model
# Stop current process và chạy lại
./run-dev.ps1

# 2. Restart recommend-service (Java) 
cd d:\LVTN\CTU-Connect-demo\recommend-service\java-api
./mvnw spring-boot:run

# 3. Restart user-service
cd d:\LVTN\CTU-Connect-demo\user-service
./mvnw spring-boot:run
```

## Thay Đổi Bổ Sung (2024-12-14)

### Fix Python Schema cho Friend Ranking
**File**: `recommend-service/python-model/server.py`

Vấn đề: Java gửi `currentUser`, `additionalScores`, `topK` (camelCase) nhưng Python expect `current_user`, `additional_scores`, `top_k` (snake_case).

Giải pháp: Tạo class `UserProfileData` và `FriendRankingRequest` với support cả 2 naming conventions:

```python
class UserProfileData(BaseModel):
    userId: Optional[str] = Field(None)  # camelCase from Java
    user_id: Optional[str] = Field(None)  # snake_case
    
    @property
    def effective_user_id(self) -> str:
        return self.userId or self.user_id or "unknown"

class FriendRankingRequest(BaseModel):
    currentUser: Optional[UserProfileData] = Field(None, alias="current_user")
    current_user: Optional[UserProfileData] = Field(None)
    
    @property
    def effective_current_user(self):
        return self.currentUser or self.current_user
```

### Bỏ qua bài viết của bản thân
**File**: `recommend-service/java-api/.../HybridRecommendationService.java`

```java
return posts.stream()
    .filter(post -> !excludePostIds.contains(post.getPostId()))
    // NEW: Exclude user's own posts
    .filter(post -> !userId.equals(post.getAuthorId()))
    .limit(limit)
    ...
```

## Frontend Integration

Nút "Làm mới" đã được implement ở `FriendSuggestions.tsx`:
- Gọi `POST /api/users/friend-suggestions/refresh`
- Server sẽ clear cache + viewed items
- Fetch fresh suggestions sau khi refresh

## Files Modified

### user-service
- `config/RedisConfig.java` - Fix default host
- `client/RecommendServiceClient.java` - Add refreshSuggestions()
- `service/SocialGraphService.java` - Add refreshFriendSuggestions()
- `controller/EnhancedUserController.java` - Update refresh endpoint

### recommend-service/java-api
- `repository/redis/RedisCacheService.java` - Add viewed tracking methods
- `service/HybridFriendRecommendationService.java` - Integrate viewed tracking
- `service/HybridRecommendationService.java` - Integrate viewed tracking
- `controller/FriendRecommendationController.java` - Add refresh endpoint
- `controller/RecommendationController.java` - Update refresh endpoint
