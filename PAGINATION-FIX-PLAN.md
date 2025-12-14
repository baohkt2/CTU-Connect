# Kế Hoạch Sửa Lỗi Phân Trang và Gợi Ý Kết Bạn

## 📋 Tổng Quan Vấn Đề

### Vấn Đề 1: HTTP Parsing Error trong Friend Suggestions
**Lỗi**: `Invalid character found in the request target [/api/users/friend-suggestions?limit=50&refresh=[object%2520Object]]`

**Nguyên Nhân**: 
- Frontend đang gửi `refresh=[object Object]` thay vì giá trị boolean hợp lệ
- Đây là lỗi từ phía client khi serialize object JavaScript thành query parameter

**Phạm Vi Ảnh Hưởng**:
- Endpoint: `GET /api/users/friend-suggestions`
- File backend: `EnhancedUserController.java`
- Tham số: `@RequestParam(defaultValue = "false") boolean refresh`

**Giải Pháp**:
1. **Frontend Fix (Ưu tiên)**:
   - Loại bỏ hoàn toàn tham số `refresh` khỏi query string
   - Thay thế nút "Refresh" bằng nút "Xem thêm" với phân trang

2. **Backend Defensive (Tạm thời)**:
   - Thay đổi từ `boolean` sang `String` và parse thủ công
   - Hoặc loại bỏ hoàn toàn tham số `refresh` khỏi endpoint

---

### Vấn Đề 2: Thiếu Phân Trang Hiệu Quả cho Friend Suggestions

**Hiện Trạng**:
```java
@GetMapping("/friend-suggestions")
public ResponseEntity<List<FriendSuggestionDTO>> getFriendSuggestions(
    @RequestParam(defaultValue = "20") int limit,
    @RequestParam(defaultValue = "false") boolean refresh) // ❌ Không có page/offset
```

**Vấn Đề**:
- Không có tham số `page` hoặc `offset`
- Mỗi lần gọi đều trả về cùng 20 kết quả từ cache
- Nút "Refresh" chỉ xóa cache, không hỗ trợ "Xem thêm"

**Giải Pháp**:
- Thêm tham số `page` hoặc `offset`
- Backend track những user ID đã được gửi cho client
- Client gửi danh sách `excludeUserIds` trong request tiếp theo
- Hoặc dùng cursor-based pagination với `lastUserId`

---

### Vấn Đề 3: Phân Trang Feed Trả Về Trùng Lặp

**Hiện Trạng**:
```java
// recommend-service/HybridRecommendationService.java
public RecommendationResponse getFeed(String userId, Integer page, Integer size) {
    // ✅ Có tham số page nhưng không được sử dụng
    
    List<RecommendedPost> cachedFeed = redisCacheService.getRecommendations(userId, ...);
    if (cachedFeed != null && !cachedFeed.isEmpty()) {
        return buildResponse(...); // ❌ Trả về toàn bộ cache, không skip theo page
    }
}
```

**Nguyên Nhân**:
1. **Cache không phân biệt page**: 
   - Cache key: `recommendations:posts:{userId}` (không có `:page`)
   - Page 0 và page 1 đều trả về cùng cache

2. **Không filter seen posts**:
   - Client nhấn "Tải thêm" → Backend trả về cache cũ
   - Không có cơ chế track `seenPostIds` từ client

3. **Khi cache miss**:
   - Backend fetch fresh posts nhưng không skip theo `page * size`
   - Kết quả: Page 1, 2, 3 đều bắt đầu từ post đầu tiên

**Giải Pháp**:
1. **Option A - Client-Sent Exclusion (Recommended)**:
   - Client track `seenPostIds` locally
   - Gửi `excludePostIds` trong request tiếp theo
   - Backend filter out các post này trước khi return

2. **Option B - Server-Side Session Tracking**:
   - Backend lưu danh sách posts đã gửi cho user trong Redis
   - TTL ngắn (5-10 phút) để avoid memory bloat
   - Mỗi request filter out các post đã gửi

3. **Option C - Cursor-Based Pagination**:
   - Client gửi `lastPostId` và `lastScore`
   - Backend fetch posts sau cursor này
   - Phù hợp với ranked feeds (có score)

---

## 🎯 Giải Pháp Chi Tiết

### Giải Pháp 1: Fix HTTP Parsing Error

#### Backend Changes
```java
// File: user-service/.../EnhancedUserController.java

// TRƯỚC:
@GetMapping("/friend-suggestions")
public ResponseEntity<List<FriendSuggestionDTO>> getFriendSuggestions(
    @RequestParam(defaultValue = "20") int limit,
    @RequestParam(defaultValue = "false") boolean refresh) { // ❌ Gây lỗi parse
    
    if (refresh) {
        socialGraphService.invalidateFriendSuggestionsCache(currentUser.getId());
    }
    // ...
}

// SAU:
@GetMapping("/friend-suggestions")
public ResponseEntity<List<FriendSuggestionDTO>> getFriendSuggestions(
    @RequestParam(defaultValue = "20") int limit,
    @RequestParam(required = false) List<String> excludeUserIds) { // ✅ Thay thế refresh
    
    // Không còn logic refresh, chỉ còn pagination với exclusion
    // ...
}
```

#### Frontend Changes
```javascript
// TRƯỚC:
const fetchSuggestions = () => {
  api.get('/api/users/friend-suggestions', {
    params: { limit: 50, refresh: { force: true } } // ❌ Object không hợp lệ
  });
}

// SAU:
const [seenUserIds, setSeenUserIds] = useState([]);

const loadMoreSuggestions = () => {
  api.get('/api/users/friend-suggestions', {
    params: { 
      limit: 20, 
      excludeUserIds: seenUserIds.join(',') // ✅ CSV string
    }
  }).then(res => {
    setSeenUserIds([...seenUserIds, ...res.data.map(u => u.userId)]);
  });
}
```

---

### Giải Pháp 2: Phân Trang Friend Suggestions

#### Backend Implementation

**File**: `user-service/.../EnhancedUserController.java`
```java
@GetMapping("/friend-suggestions")
@RequireAuth
public ResponseEntity<Map<String, Object>> getFriendSuggestions(
        @RequestParam(defaultValue = "20") int limit,
        @RequestParam(required = false) String excludeUserIds) { // CSV: "id1,id2,id3"
    
    AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
    
    // Parse excluded user IDs
    Set<String> excludedIds = new HashSet<>();
    if (excludeUserIds != null && !excludeUserIds.isEmpty()) {
        excludedIds.addAll(Arrays.asList(excludeUserIds.split(",")));
    }
    
    // Get suggestions and filter out excluded ones
    List<FriendSuggestionDTO> allSuggestions = 
        socialGraphService.getFriendSuggestions(currentUser.getId(), limit * 3);
    
    List<FriendSuggestionDTO> filteredSuggestions = allSuggestions.stream()
        .filter(s -> !excludedIds.contains(s.getUserId()))
        .limit(limit)
        .collect(Collectors.toList());
    
    boolean hasMore = allSuggestions.size() > (excludedIds.size() + limit);
    
    return ResponseEntity.ok(Map.of(
        "suggestions", filteredSuggestions,
        "hasMore", hasMore,
        "total", allSuggestions.size()
    ));
}
```

**File**: `user-service/.../SocialGraphService.java`
```java
// Tăng limit để over-fetch (vì sẽ filter ở controller)
public List<FriendSuggestionDTO> getFriendSuggestions(String userId, int limit) {
    String cacheKey = "friend_suggestions:" + userId;
    
    // Cache vẫn lưu large list
    List<FriendSuggestionDTO> cached = 
        (List<FriendSuggestionDTO>) redisTemplate.opsForValue().get(cacheKey);
    
    if (cached != null && !cached.isEmpty()) {
        return cached.stream().limit(limit).collect(Collectors.toList());
    }
    
    // Fetch large list from recommend-service (50-100 suggestions)
    List<FriendSuggestionDTO> suggestions = 
        recommendServiceClient.getMLFriendSuggestions(userId, 100); // Over-fetch
    
    // Cache for 8 hours
    redisTemplate.opsForValue().set(cacheKey, suggestions, 8, TimeUnit.HOURS);
    
    return suggestions.stream().limit(limit).collect(Collectors.toList());
}
```

---

### Giải Pháp 3: Fix Feed Pagination Duplicates

#### Phương Pháp: Client-Sent Exclusion List

**Tại Sao Chọn Phương Pháp Này?**
- ✅ Đơn giản, không cần server-side session storage
- ✅ Stateless, dễ scale
- ✅ Client control trải nghiệm (có thể refresh để xem lại posts cũ)
- ❌ Trade-off: Request size tăng khi user scroll nhiều

**Khi Nào Cần Phương Pháp Khác?**
- Nếu `excludePostIds` quá lớn (>1000 posts) → Dùng cursor-based
- Nếu cần "infinite scroll" không giới hạn → Server-side tracking

#### Backend Changes

**File**: `recommend-service/.../RecommendationController.java`
```java
@GetMapping("/feed/{userId}")
public ResponseEntity<RecommendationResponse> getUserFeed(
        @PathVariable String userId,
        @RequestParam(defaultValue = "0") Integer page,
        @RequestParam(defaultValue = "20") Integer size,
        @RequestParam(required = false) String excludePostIds) { // NEW: CSV "postId1,postId2,..."
    
    log.info("📥 GET /feed/{} - Page: {}, Size: {}, Exclude: {} posts", 
        userId, page, size, 
        excludePostIds != null ? excludePostIds.split(",").length : 0);
    
    // Parse excluded post IDs
    Set<String> excludedIds = new HashSet<>();
    if (excludePostIds != null && !excludePostIds.isEmpty()) {
        excludedIds.addAll(Arrays.asList(excludePostIds.split(",")));
    }
    
    RecommendationResponse response = recommendationService.getFeed(
        userId, page, size, excludedIds); // Pass exclusion set
    
    return ResponseEntity.ok(response);
}
```

**File**: `recommend-service/.../HybridRecommendationService.java`
```java
public RecommendationResponse getFeed(String userId, Integer page, Integer size, 
                                       Set<String> excludePostIds) {
    long startTime = System.currentTimeMillis();
    
    int requestSize = size != null ? size : defaultRecommendationCount;
    
    // ❌ REMOVE: Cache check (vì cache không track exclusion)
    // ✅ ALWAYS fetch fresh recommendations and filter
    
    // Get user profile and interaction history
    UserAcademicProfile userProfile = userServiceClient.getUserAcademicProfile(userId);
    List<UserInteractionHistory> userHistory = getUserInteractionHistory(userId, 30);
    
    // Combine exclusions: history + client-sent
    Set<String> allExcludedIds = new HashSet<>(excludePostIds);
    allExcludedIds.addAll(userHistory.stream()
        .map(UserInteractionHistory::getPostId)
        .collect(Collectors.toSet()));
    
    log.info("🚫 Excluding {} posts ({} from history, {} from client)", 
        allExcludedIds.size(),
        userHistory.size(),
        excludePostIds.size());
    
    // Get candidate posts (exclude seen + excluded)
    List<CandidatePost> candidatePosts = getCandidatePosts(userId, allExcludedIds, requestSize * 5);
    
    if (candidatePosts.isEmpty()) {
        log.warn("No new posts available for user: {}", userId);
        return buildEmptyResponse(userId, startTime);
    }
    
    // ML ranking
    List<RecommendedPost> finalRecommendations;
    if (pythonServiceEnabled) {
        PythonModelRequest modelRequest = PythonModelRequest.builder()
            .userAcademic(userProfile)
            .userHistory(userHistory)
            .candidatePosts(candidatePosts)
            .topK(requestSize * 2)
            .build();
        
        PythonModelResponse modelResponse = pythonModelService.predictRanking(modelRequest);
        finalRecommendations = convertPythonResponse(modelResponse, candidatePosts);
    } else {
        finalRecommendations = fallbackRanking(candidatePosts, requestSize);
    }
    
    // Apply business rules
    finalRecommendations = applyBusinessRules(userId, finalRecommendations, userProfile);
    
    // Limit to requested size
    finalRecommendations = finalRecommendations.stream()
        .limit(requestSize)
        .collect(Collectors.toList());
    
    // ✅ NEW: Don't cache (because each request has different exclusions)
    // Or cache with exclusion list as part of key (complex, not recommended)
    
    return buildResponse(userId, finalRecommendations, requestSize, startTime, "fresh");
}
```

**File**: `recommend-service/.../HybridRecommendationService.java` (Helper)
```java
private List<CandidatePost> getCandidatePosts(String userId, Set<String> excludePostIds, int limit) {
    log.debug("Fetching candidate posts (limit: {}, exclude: {})", limit, excludePostIds.size());
    
    // Fetch posts from database
    List<PostDTO> allPosts = postServiceClient.getRecentPosts(limit * 2);
    
    return allPosts.stream()
        .filter(post -> !excludePostIds.contains(post.getPostId())) // ✅ Filter excluded
        .filter(post -> !userId.equals(post.getAuthorId())) // ✅ Exclude user's own posts
        .limit(limit)
        .map(this::convertToCandidatePost)
        .collect(Collectors.toList());
}
```

#### post-service Changes

**File**: `post-service/.../PostController.java`
```java
@GetMapping("/feed")
@RequireAuth
public ResponseEntity<?> getPersonalizedFeed(
        @RequestParam(defaultValue = "0") int page,
        @RequestParam(defaultValue = "10") int size,
        @RequestParam(required = false) String excludePostIds) { // NEW
    
    String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
    
    log.info("📥 GET /api/posts/feed - User: {}, Page: {}, Size: {}, Exclude: {} posts", 
        currentUserId, page, size,
        excludePostIds != null ? excludePostIds.split(",").length : 0);
    
    // Call recommendation-service with exclusion list
    if (recommendationServiceClient != null) {
        try {
            RecommendationFeedResponse recommendationResponse = 
                recommendationServiceClient.getRecommendationFeed(
                    currentUserId, page, size, excludePostIds); // Pass through
            
            // ... rest of logic
        } catch (Exception e) {
            log.error("Recommendation service error: {}", e.getMessage());
        }
    }
    
    // Fallback: trending posts
    List<PostResponse> posts = newsFeedService.getTrendingPosts(page, size);
    return ResponseEntity.ok(posts);
}
```

**File**: `post-service/.../RecommendationServiceClient.java`
```java
@FeignClient(name = "recommend-service", url = "${services.recommend-service.url}")
public interface RecommendationServiceClient {
    
    @GetMapping("/api/recommendations/feed/{userId}")
    RecommendationFeedResponse getRecommendationFeed(
            @PathVariable("userId") String userId,
            @RequestParam(value = "page", defaultValue = "0") Integer page,
            @RequestParam(value = "size", defaultValue = "10") Integer size,
            @RequestParam(value = "excludePostIds", required = false) String excludePostIds); // NEW
}
```

---

## 📊 Tóm Tắt Thay Đổi

### Backend Files to Modify

| Service | File | Changes |
|---------|------|---------|
| **user-service** | `EnhancedUserController.java` | Remove `refresh` param, add `excludeUserIds` param, return `hasMore` flag |
| **user-service** | `SocialGraphService.java` | Over-fetch suggestions (100 instead of 20) for filtering |
| **recommend-service** | `RecommendationController.java` | Add `excludePostIds` param to `/feed/{userId}` |
| **recommend-service** | `HybridRecommendationService.java` | Remove cache check for feeds, filter by `excludePostIds` |
| **post-service** | `PostController.java` | Add `excludePostIds` param to `/feed` |
| **post-service** | `RecommendationServiceClient.java` | Add `excludePostIds` param to Feign client |

### Frontend Changes (Not in Scope but Important)

```javascript
// Friend Suggestions
const [seenUserIds, setSeenUserIds] = useState([]);
const [hasMore, setHasMore] = useState(true);

const loadMoreSuggestions = () => {
  api.get('/api/users/friend-suggestions', {
    params: { 
      limit: 20,
      excludeUserIds: seenUserIds.join(',')
    }
  }).then(res => {
    setSuggestions([...suggestions, ...res.data.suggestions]);
    setSeenUserIds([...seenUserIds, ...res.data.suggestions.map(u => u.userId)]);
    setHasMore(res.data.hasMore);
  });
};

// Feed
const [seenPostIds, setSeenPostIds] = useState([]);

const loadMorePosts = () => {
  api.get('/api/posts/feed', {
    params: {
      page: 0, // Always 0, exclusion list handles pagination
      size: 10,
      excludePostIds: seenPostIds.join(',')
    }
  }).then(res => {
    setPosts([...posts, ...res.data]);
    setSeenPostIds([...seenPostIds, ...res.data.map(p => p.postId)]);
  });
};
```

---

## ⚠️ Trade-offs và Cân Nhắc

### Client-Sent Exclusion List

**Ưu Điểm**:
- ✅ Stateless backend, dễ scale horizontally
- ✅ Không cần Redis/database để track sessions
- ✅ Client có thể refresh để xem lại posts cũ

**Nhược Điểm**:
- ❌ Request size tăng theo số post đã xem (có thể lên đến 10KB+ sau 100 posts)
- ❌ URL length limit (2048 chars cho GET) có thể bị vượt nếu dùng GET

**Giải Pháp Cho URL Length**:
```java
// Chuyển từ GET sang POST nếu excludePostIds quá dài
@PostMapping("/feed")
public ResponseEntity<RecommendationResponse> getUserFeed(
        @RequestBody FeedRequest request) {
    // request.userId, request.page, request.size, request.excludePostIds (List<String>)
}
```

### Alternative: Server-Side Session Tracking

**Khi Nào Dùng**:
- User có thể scroll qua hàng trăm/nghìn posts
- Không muốn client phải gửi large exclusion list

**Implementation**:
```java
// Redis key: "feed_session:{userId}:{sessionId}"
// Value: Set<String> of sent post IDs
// TTL: 10 minutes (expire nếu user không scroll tiếp)

@GetMapping("/feed/{userId}")
public ResponseEntity<RecommendationResponse> getUserFeed(
        @PathVariable String userId,
        @RequestParam(required = false) String sessionId) {
    
    String session = sessionId != null ? sessionId : UUID.randomUUID().toString();
    String redisKey = "feed_session:" + userId + ":" + session;
    
    // Get already-sent post IDs from Redis
    Set<String> sentPostIds = redisTemplate.opsForSet().members(redisKey);
    
    // Get fresh recommendations excluding sent posts
    List<RecommendedPost> posts = getFreshPosts(userId, sentPostIds, 20);
    
    // Track sent posts
    redisTemplate.opsForSet().add(redisKey, 
        posts.stream().map(p -> p.getPostId()).toArray(String[]::new));
    redisTemplate.expire(redisKey, 10, TimeUnit.MINUTES);
    
    return ResponseEntity.ok(new RecommendationResponse(posts, session));
}
```

---

## 🚀 Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. ✅ Remove `refresh` param from friend suggestions endpoint
2. ✅ Add `excludeUserIds` param to friend suggestions
3. ✅ Fix HTTP parsing error

### Phase 2: Feed Pagination (High Priority)
1. ✅ Add `excludePostIds` param to feed endpoints
2. ✅ Remove cache check for paginated feeds
3. ✅ Filter posts by exclusion list in `getCandidatePosts()`

### Phase 3: Frontend Updates (Required for Complete Fix)
1. Track `seenUserIds` and `seenPostIds` in React state
2. Change "Refresh" button to "Load More" button
3. Append new results instead of replacing

### Phase 4: Optimization (Optional)
1. Implement Redis session tracking for feeds (if needed)
2. Add cursor-based pagination for very long feeds
3. Add analytics for scroll depth and suggestion quality

---

## 🧪 Testing Checklist

### Friend Suggestions
- [ ] Call `/friend-suggestions?limit=20` → Get 20 suggestions
- [ ] Call `/friend-suggestions?limit=20&excludeUserIds=id1,id2` → Get 20 NEW suggestions
- [ ] Call 3rd time → Verify no duplicates
- [ ] Clear cache → Verify can see old suggestions again if not excluded

### Feed Pagination
- [ ] Load feed page 1 → Get 10 posts
- [ ] Load more with `excludePostIds` → Get 10 NEW posts
- [ ] Scroll 5 times → Verify 50 unique posts, no duplicates
- [ ] Refresh page → Can see old posts again (fresh session)

### Search Compatibility
- [ ] Search for users → Works without `excludeUserIds`
- [ ] Search then load more → Filters work correctly
- [ ] Mix search and suggestions → No conflicts

---

## 📝 Migration Notes

### Breaking Changes
- ❌ `refresh=true` param removed from friend suggestions
- ⚠️ Frontend MUST update to use `excludeUserIds` for load more
- ⚠️ Feed responses will be different (no more duplicates)

### Backward Compatibility
- ✅ If `excludeUserIds` is omitted → Works like before
- ✅ Existing cache will expire naturally (8h TTL)
- ✅ No database migrations required

### Rollback Plan
- Keep old endpoints for 1 week with deprecation warning
- Monitor error logs for clients still using `refresh=true`
- Gradual cutover: 10% → 50% → 100% users
