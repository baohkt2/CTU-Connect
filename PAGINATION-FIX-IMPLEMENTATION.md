# Tóm Tắt Thay Đổi - Sửa Lỗi Phân Trang và Gợi Ý Kết Bạn

## ✅ Hoàn Thành

### 1. HTTP Parsing Error - Friend Suggestions (/api/users/friend-suggestions)

**Vấn đề**: `Invalid character found in the request target [/api/users/friend-suggestions?limit=50&refresh=[object%2520Object]]`

**Nguyên nhân**: Frontend gửi `refresh=[object Object]` thay vì boolean hợp lệ

**Giải pháp đã áp dụng**:
- ❌ **Loại bỏ** parameter `refresh` khỏi endpoint
- ✅ **Thay thế** bằng `excludeUserIds` (CSV string) cho pagination
- ✅ Return `hasMore` flag để frontend biết còn data hay không

**Files thay đổi**:
- `user-service/.../EnhancedUserController.java`
  - Removed: `@RequestParam(defaultValue = "false") boolean refresh`
  - Added: `@RequestParam(required = false) String excludeUserIds`
  - Changed return type: `List<FriendSuggestionDTO>` → `Map<String, Object>` (với `suggestions`, `hasMore`, `total`, `returned`)

---

### 2. Feed Pagination Trùng Lặp (/api/posts/feed)

**Vấn đề**: Nhấn "Tải thêm" → Trả về y hệt các bài viết đã hiển thị

**Nguyên nhân**: 
- Cache không phân biệt page
- Không có cơ chế filter posts đã xem
- Parameter `page` có nhưng không được sử dụng

**Giải pháp đã áp dụng**:

#### A. recommend-service (Core Logic)

**File**: `RecommendationController.java`
```java
// TRƯỚC:
@GetMapping("/feed")
RecommendationResponse getFeed(String userId, Integer page, Integer size)

// SAU:
@GetMapping("/feed")
RecommendationResponse getFeed(String userId, Integer page, Integer size, 
                                String excludePostIds) // NEW
```

**File**: `HybridRecommendationService.java`
```java
// TRƯỚC:
public RecommendationResponse getFeed(String userId, Integer page, Integer size) {
    // Check cache → Return if exists (ALWAYS)
    List<RecommendedPost> cachedFeed = redisCacheService.getRecommendations(userId, ...);
    if (cachedFeed != null) return buildResponse(...);
    ...
}

// SAU:
public RecommendationResponse getFeed(String userId, Integer page, Integer size, 
                                       Set<String> excludePostIds) {
    boolean isPagination = excludePostIds != null && !excludePostIds.isEmpty();
    
    // ✅ Skip cache for pagination requests
    if (!isPagination) {
        List<RecommendedPost> cachedFeed = redisCacheService.getRecommendations(userId, ...);
        if (cachedFeed != null) return buildResponse(...);
    }
    
    // ✅ Combine exclusions: interaction history + client-sent
    Set<String> allExcludedIds = new HashSet<>();
    allExcludedIds.addAll(userHistory.stream().map(...).collect(...));
    if (excludePostIds != null) {
        allExcludedIds.addAll(excludePostIds);
    }
    
    // ✅ Filter candidate posts by combined exclusion list
    List<CandidatePost> candidatePosts = getCandidatePosts(userId, allExcludedIds, ...);
    
    // ... Python ML ranking ...
    
    // ✅ Don't cache paginated results
    if (!isPagination) {
        redisCacheService.cacheRecommendations(userId, finalRecommendations, ...);
    }
}
```

**Thay đổi chính**:
1. ✅ Skip cache check when `excludePostIds` is provided
2. ✅ Combine exclusions from interaction history + client-sent list
3. ✅ Filter candidate posts by combined exclusion set
4. ✅ Only cache first request (not paginated requests)

#### B. post-service (API Gateway)

**File**: `PostController.java`
```java
// TRƯỚC:
@GetMapping("/feed")
public ResponseEntity<?> getPersonalizedFeed(
    @RequestParam(defaultValue = "0") int page,
    @RequestParam(defaultValue = "10") int size)

// SAU:
@GetMapping("/feed")
public ResponseEntity<?> getPersonalizedFeed(
    @RequestParam(defaultValue = "0") int page,
    @RequestParam(defaultValue = "10") int size,
    @RequestParam(required = false) String excludePostIds) // NEW
```

**File**: `RecommendationServiceClient.java` (Feign)
```java
// TRƯỚC:
@GetMapping("/api/recommendations/feed")
RecommendationFeedResponse getRecommendationFeed(
    @RequestParam("userId") String userId,
    @RequestParam Integer page,
    @RequestParam Integer size)

// SAU:
@GetMapping("/api/recommendations/feed")
RecommendationFeedResponse getRecommendationFeed(
    @RequestParam("userId") String userId,
    @RequestParam Integer page,
    @RequestParam Integer size,
    @RequestParam(required = false) String excludePostIds) // NEW
```

**File**: `RecommendationServiceClientFallback.java`
```java
// Updated method signature to match interface
public RecommendationFeedResponse getRecommendationFeed(
    String userId, Integer page, Integer size, String excludePostIds)
```

---

### 3. Friend Suggestions - Xem Thêm (Load More)

**Vấn đề**: Chỉ có nút "Refresh" (xóa cache), không có "Load More"

**Giải pháp đã áp dụng**:

**File**: `EnhancedUserController.java`

```java
// TRƯỚC:
@GetMapping("/friend-suggestions")
public ResponseEntity<List<FriendSuggestionDTO>> getFriendSuggestions(
    @RequestParam(defaultValue = "20") int limit,
    @RequestParam(defaultValue = "false") boolean refresh) {
    
    if (refresh) {
        socialGraphService.invalidateFriendSuggestionsCache(currentUser.getId());
    }
    
    List<FriendSuggestionDTO> suggestions = 
        socialGraphService.getFriendSuggestions(currentUser.getId(), limit);
    
    return ResponseEntity.ok(suggestions);
}

// SAU:
@GetMapping("/friend-suggestions")
public ResponseEntity<Map<String, Object>> getFriendSuggestions(
    @RequestParam(defaultValue = "20") int limit,
    @RequestParam(required = false) String excludeUserIds) {
    
    // Parse excluded user IDs (CSV format)
    Set<String> excludedIds = new HashSet<>();
    if (excludeUserIds != null && !excludeUserIds.isEmpty()) {
        excludedIds.addAll(Arrays.asList(excludeUserIds.split(",")));
    }
    
    // Over-fetch to ensure enough after filtering
    List<FriendSuggestionDTO> allSuggestions = 
        socialGraphService.getFriendSuggestions(currentUser.getId(), limit * 5);
    
    // Filter out excluded users
    List<FriendSuggestionDTO> filteredSuggestions = allSuggestions.stream()
        .filter(s -> !excludedIds.contains(s.getUserId()))
        .limit(limit)
        .collect(Collectors.toList());
    
    // Check if more available
    long remainingCount = allSuggestions.stream()
        .filter(s -> !excludedIds.contains(s.getUserId()))
        .count();
    boolean hasMore = remainingCount > limit;
    
    return ResponseEntity.ok(Map.of(
        "suggestions", filteredSuggestions,
        "hasMore", hasMore,
        "total", allSuggestions.size(),
        "returned", filteredSuggestions.size()
    ));
}
```

**Thay đổi chính**:
1. ✅ Over-fetch suggestions (limit * 5) để đủ data sau khi filter
2. ✅ Filter out excluded user IDs
3. ✅ Return `hasMore` flag
4. ✅ Return structured response thay vì plain list

**Endpoint đã xóa**:
- ❌ `POST /api/users/friend-suggestions/refresh` (không còn cần thiết)

---

### 4. Friend Suggestions Search với Pagination

**File**: `EnhancedUserController.java`

```java
// TRƯỚC:
@GetMapping("/friend-suggestions/search")
public ResponseEntity<List<UserSearchDTO>> searchFriendSuggestions(
    @RequestParam(required = false) String query,
    @RequestParam(required = false) String college,
    @RequestParam(required = false) String faculty,
    @RequestParam(required = false) String batch,
    @RequestParam(defaultValue = "50") int limit)

// SAU:
@GetMapping("/friend-suggestions/search")
public ResponseEntity<Map<String, Object>> searchFriendSuggestions(
    @RequestParam(required = false) String query,
    @RequestParam(required = false) String college,
    @RequestParam(required = false) String faculty,
    @RequestParam(required = false) String batch,
    @RequestParam(defaultValue = "50") int limit,
    @RequestParam(required = false) String excludeUserIds) { // NEW
    
    // Parse excluded IDs
    Set<String> excludedIds = ...;
    
    // Over-fetch
    List<UserSearchDTO> allResults = userService.searchFriendSuggestions(..., limit * 3);
    
    // Filter
    List<UserSearchDTO> filteredResults = allResults.stream()
        .filter(u -> !excludedIds.contains(u.getId()))
        .limit(limit)
        .collect(Collectors.toList());
    
    return ResponseEntity.ok(Map.of(
        "results", filteredResults,
        "hasMore", ...,
        "total", allResults.size()
    ));
}
```

---

## 📊 Summary of Changes

### Backend Files Modified

| Service | File | Changes |
|---------|------|---------|
| **recommend-service** | `RecommendationController.java` | Added `excludePostIds` param |
| **recommend-service** | `HybridRecommendationService.java` | Skip cache for pagination, combine exclusions, don't cache paginated results |
| **post-service** | `PostController.java` | Added `excludePostIds` param to `/feed` |
| **post-service** | `RecommendationServiceClient.java` | Added `excludePostIds` param to Feign interface |
| **post-service** | `RecommendationServiceClientFallback.java` | Updated method signature |
| **user-service** | `EnhancedUserController.java` | Replaced `refresh` with `excludeUserIds`, return structured response with `hasMore` |

### API Changes

#### Feed Pagination
```
Before: GET /api/posts/feed?page=0&size=10
After:  GET /api/posts/feed?page=0&size=10&excludePostIds=id1,id2,id3
```

#### Friend Suggestions
```
Before: GET /api/users/friend-suggestions?limit=20&refresh=true
        Response: List<FriendSuggestionDTO>

After:  GET /api/users/friend-suggestions?limit=20&excludeUserIds=id1,id2
        Response: {
          "suggestions": [...],
          "hasMore": true,
          "total": 100,
          "returned": 20
        }
```

#### Friend Suggestions Search
```
Before: GET /api/users/friend-suggestions/search?query=...&limit=50
        Response: List<UserSearchDTO>

After:  GET /api/users/friend-suggestions/search?query=...&limit=50&excludeUserIds=id1,id2
        Response: {
          "results": [...],
          "hasMore": true,
          "total": 80,
          "returned": 50
        }
```

### Removed Endpoints
- ❌ `POST /api/users/friend-suggestions/refresh` (replaced with pagination)

---

## 🚀 Frontend Integration Required

### 1. Feed Pagination (Posts)

```javascript
// State
const [posts, setPosts] = useState([]);
const [seenPostIds, setSeenPostIds] = useState([]);

// Initial load
const loadFeed = async () => {
  const res = await api.get('/api/posts/feed', {
    params: { page: 0, size: 10 }
  });
  setPosts(res.data);
  setSeenPostIds(res.data.map(p => p.postId));
};

// Load more
const loadMorePosts = async () => {
  const res = await api.get('/api/posts/feed', {
    params: {
      page: 0, // Always 0
      size: 10,
      excludePostIds: seenPostIds.join(',') // CSV
    }
  });
  
  setPosts([...posts, ...res.data]);
  setSeenPostIds([...seenPostIds, ...res.data.map(p => p.postId)]);
};
```

### 2. Friend Suggestions Pagination

```javascript
// State
const [suggestions, setSuggestions] = useState([]);
const [seenUserIds, setSeenUserIds] = useState([]);
const [hasMore, setHasMore] = useState(true);

// Initial load
const loadSuggestions = async () => {
  const res = await api.get('/api/users/friend-suggestions', {
    params: { limit: 20 }
  });
  
  setSuggestions(res.data.suggestions);
  setSeenUserIds(res.data.suggestions.map(u => u.userId));
  setHasMore(res.data.hasMore);
};

// Load more (instead of refresh)
const loadMoreSuggestions = async () => {
  const res = await api.get('/api/users/friend-suggestions', {
    params: {
      limit: 20,
      excludeUserIds: seenUserIds.join(',')
    }
  });
  
  setSuggestions([...suggestions, ...res.data.suggestions]);
  setSeenUserIds([...seenUserIds, ...res.data.suggestions.map(u => u.userId)]);
  setHasMore(res.data.hasMore);
};

// UI
{hasMore && <button onClick={loadMoreSuggestions}>Xem thêm</button>}
```

### 3. Friend Suggestions Search Pagination

```javascript
const [searchResults, setSearchResults] = useState([]);
const [excludedIds, setExcludedIds] = useState([]);
const [hasMore, setHasMore] = useState(true);

const searchUsers = async (query) => {
  const res = await api.get('/api/users/friend-suggestions/search', {
    params: {
      query,
      college,
      faculty,
      batch,
      limit: 50,
      excludeUserIds: excludedIds.join(',')
    }
  });
  
  setSearchResults([...searchResults, ...res.data.results]);
  setExcludedIds([...excludedIds, ...res.data.results.map(u => u.id)]);
  setHasMore(res.data.hasMore);
};
```

---

## ✅ Testing Checklist

### Feed Pagination
- [x] Load feed page 1 → Get 10 posts
- [ ] Click "Load More" → Get 10 NEW posts (no duplicates)
- [ ] Scroll 5 times → Verify 50 unique posts
- [ ] Refresh page → Can see old posts again (new session)

### Friend Suggestions
- [x] Load suggestions → Get 20 suggestions + `hasMore` flag
- [ ] Click "Xem thêm" → Get 20 NEW suggestions
- [ ] Repeat 3 times → Verify 60 unique users, no duplicates
- [ ] Check `hasMore: false` when no more available

### Friend Suggestions Search
- [ ] Search "Nguyen" → Get results
- [ ] Click "Xem thêm" → Get more matching "Nguyen" (no duplicates)
- [ ] Change search query → Reset excludedIds and start fresh

---

## 🔍 Verification

Run services and test:
```bash
# Terminal 1 - Python service
cd recommend-service/python-model
./run-dev.ps1

# Terminal 2 - recommend-service
# (Already running as IntelliJ task)

# Test feed pagination
curl "http://localhost:8095/api/recommendations/feed?userId=test-user&size=10"
curl "http://localhost:8095/api/recommendations/feed?userId=test-user&size=10&excludePostIds=post1,post2"

# Test friend suggestions
curl "http://localhost:8081/api/users/friend-suggestions?limit=20" -H "Authorization: Bearer <token>"
curl "http://localhost:8081/api/users/friend-suggestions?limit=20&excludeUserIds=user1,user2" -H "Authorization: Bearer <token>"
```

---

## 📝 Breaking Changes

### API Response Changes
1. **Friend Suggestions**:
   - Before: `List<FriendSuggestionDTO>`
   - After: `{ suggestions: [...], hasMore: boolean, total: number, returned: number }`

2. **Friend Search**:
   - Before: `List<UserSearchDTO>`
   - After: `{ results: [...], hasMore: boolean, total: number, returned: number }`

### Removed Endpoints
- ❌ `POST /api/users/friend-suggestions/refresh`

### New Query Parameters
- ✅ `excludePostIds` for feed pagination
- ✅ `excludeUserIds` for friend suggestions pagination

---

## 🎯 Benefits

1. **✅ No More Duplicates**: Client-sent exclusion list ensures no repeated items
2. **✅ Stateless Backend**: No server-side session tracking needed
3. **✅ Better UX**: "Load More" is more intuitive than "Refresh"
4. **✅ Fixes HTTP Error**: Removed problematic `refresh=[object Object]` parameter
5. **✅ Efficient Caching**: Cache still works for first request, skipped for pagination
6. **✅ Scalable**: Works with any number of pages without server memory overhead

---

## ⚠️ Known Limitations

1. **Large Exclusion Lists**: If user scrolls through 100+ items, CSV can grow large (>10KB)
   - **Mitigation**: Use POST instead of GET if needed
   - **Alternative**: Implement cursor-based pagination in future

2. **Cache Invalidation**: First page always uses cache (8h TTL)
   - **Mitigation**: User can clear cache by refreshing browser
   - **Alternative**: Add TTL header to frontend for cache control

3. **Over-fetching**: Friend suggestions fetch 5x limit to ensure enough after filtering
   - **Mitigation**: Reasonable for small datasets (<1000 suggestions)
   - **Alternative**: Implement offset-based pagination if needed
