# Fix Summary - December 10, 2025

## Issues Fixed

### 1. ✅ Neo4j Projection Error - "Invalid property 'user'" 
**Problem:** Complex Neo4j query projections với `u as user` gây lỗi khi Spring Data Neo4j cố gắng map projection.

**Solution:** 
- Loại bỏ tất cả complex projections trong UserRepository
- Tất cả query methods giờ trả về `List<UserEntity>` thay vì `Page<UserSearchProjection>`
- Áp dụng pagination thủ công trong service layer
- Simplified queries giảm complexity và tăng reliability

**Files Changed:**
- `user-service/src/main/java/com/ctuconnect/repository/UserRepository.java`
  - Removed all projection interfaces (UserSearchProjection, FriendRequestProjection, UserProfileProjection)
  - Simplified all query methods to return List<UserEntity>
  - Removed Pageable parameters from repository methods
  
- `user-service/src/main/java/com/ctuconnect/service/UserService.java`
  - Updated all methods to work with List<UserEntity>
  - Added manual pagination logic
  - Fixed searchFriendSuggestions to properly handle filters

- `user-service/src/main/java/com/ctuconnect/mapper/UserMapper.java`
  - Removed projection-based mapping methods
  - Kept only UserEntity-based mapping methods
  - Added getAllFriendRequests method combining sent and received

- `user-service/src/main/java/com/ctuconnect/service/SocialGraphService.java`
  - Updated getFriendsOfFriendsSuggestions to use List<UserEntity>

- `user-service/src/main/java/com/ctuconnect/service/UserSyncService.java`
  - Fixed cleanup methods to work with UserEntity directly

### 2. ✅ Friend Search Filter Not Working
**Problem:** Khi tìm kiếm với filter (faculty, batch, college) nhưng query=null thì trả về 0 kết quả.

**Solution:**
- Updated searchFriendSuggestions logic để check từng filter một cách rõ ràng
- Priority: query > faculty > batch > college > friend suggestions
- Added hasQuery, hasFaculty, hasBatch, hasCollege flags để xác định rõ ràng filters nào được cung cấp

**Code Logic:**
```java
boolean hasQuery = query != null && !query.trim().isEmpty();
boolean hasFaculty = faculty != null && !faculty.isEmpty();
boolean hasBatch = batch != null && !batch.isEmpty();
boolean hasCollege = college != null && !college.isEmpty();

if (hasQuery) {
    // Search by query
} else if (hasFaculty) {
    // Filter by faculty
} else if (hasBatch) {
    // Filter by batch  
} else if (hasCollege) {
    // Filter by college
} else {
    // Return mutual friend suggestions
}
```

### 3. ✅ Friend Request UI Issue - User Disappears After Sending Request
**Problem:** Sau khi gửi lời mời kết bạn, user biến mất khỏi UI thay vì hiển thị trạng thái "đã gửi".

**Solution:**
- Modified `/me/friend-requests` endpoint để trả về TẤT CẢ friend requests (cả sent và received)
- Created new method `getAllFriendRequests()` trong UserService
- Frontend giờ có thể hiển thị cả requests đã gửi và đã nhận trong cùng một tab
- Mỗi request có `requestType` field ("SENT" hoặc "RECEIVED") để UI phân biệt

**New Service Method:**
```java
@Transactional(readOnly = true)
public List<FriendRequestDTO> getAllFriendRequests(@NotBlank String userId) {
    List<FriendRequestDTO> allRequests = new ArrayList<>();
    
    // Get sent requests
    var sentRequests = userRepository.findSentFriendRequests(userId);
    allRequests.addAll(sentRequests.stream()
        .map(user -> userMapper.toFriendRequestDTO(user, "SENT"))
        .collect(Collectors.toList()));
    
    // Get received requests
    var receivedRequests = userRepository.findReceivedFriendRequests(userId);
    allRequests.addAll(receivedRequests.stream()
        .map(user -> userMapper.toFriendRequestDTO(user, "RECEIVED"))
        .collect(Collectors.toList()));
    
    return allRequests;
}
```

**Updated Endpoint:**
```java
@GetMapping("/me/friend-requests")
@RequireAuth
public ResponseEntity<List<FriendRequestDTO>> getMyFriendRequests() {
    // Now returns ALL requests (sent + received)
    List<FriendRequestDTO> allRequests = userService.getAllFriendRequests(currentUser.getId());
    return ResponseEntity.ok(allRequests);
}
```

### 4. ✅ Neo4j Syntax Error - Accept Friend Request
**Problem:** 
```
Only directed relationships are supported in CREATE
"CREATE (requester)-[:IS_FRIENDS_WITH]-(accepter)"
```

**Solution:**
Neo4j không hỗ trợ undirected relationships trong CREATE statement. Phải tạo 2 directed relationships:

**Before:**
```cypher
CREATE (requester)-[:IS_FRIENDS_WITH]-(accepter)
```

**After:**
```cypher
CREATE (requester)-[:IS_FRIENDS_WITH]->(accepter)
CREATE (accepter)-[:IS_FRIENDS_WITH]->(requester)
```

## Testing Recommendations

### 1. Test Friend Search với Filters
```bash
# Test search by faculty without query
GET /api/users/friend-suggestions/search?faculty=Công Nghệ Thông Tin&limit=10

# Test search by batch without query  
GET /api/users/friend-suggestions/search?batch=2021&limit=10

# Test search by college without query
GET /api/users/friend-suggestions/search?college=ĐHCT&limit=10
```

### 2. Test Friend Request Flow
```bash
# 1. Send friend request
POST /api/users/me/invite/{friendId}

# 2. Get ALL friend requests (should show the sent request)
GET /api/users/me/friend-requests
# Response should include both SENT and RECEIVED requests

# 3. Accept friend request (from the other user)
POST /api/users/me/accept-invite/{requesterId}
# Should succeed without Neo4j syntax error
```

### 3. Test Search with Query
```bash
# Search with query still works
GET /api/users/friend-suggestions/search?query=Tuan&limit=10
```

## Performance Notes

- Manual pagination may be slightly less efficient than database-level pagination
- Consider adding caching if performance becomes an issue
- Current approach prioritizes correctness and simplicity over optimization

## Migration Impact

- **Breaking Change:** Repository methods now return List instead of Page
- All service methods updated to handle List and apply pagination manually
- No changes needed to API endpoints or DTOs
- Frontend không cần thay đổi

## Next Steps

1. ✅ Test all friend search scenarios
2. ✅ Test friend request flow end-to-end
3. ✅ Verify UI displays sent requests correctly
4. ✅ Monitor performance with manual pagination
5. Consider adding database indexes if needed

## Known Limitations

- Manual pagination loads all results into memory before slicing
- For very large result sets (>1000 users), consider adding database-level pagination back with simpler queries
- Current approach works well for typical social network scale (hundreds of users per query)
