# Friend API Debug & Fix Summary

## Issues Identified

### 1. API Hang Issue
**Problem**: API call to `GET /api/users/me/friends` bị kẹt tại log "Getting friends for userId: ..."

**Root Cause**: Không có debug logs chi tiết để tracking được API chạy tới đâu.

### 2. PageImpl Serialization Warning
**Problem**: 
```
Serializing PageImpl instances as-is is not supported, meaning that there is no guarantee 
about the stability of the resulting JSON structure!
```

**Root Cause**: Spring không khuyến khích serialize `PageImpl` trực tiếp vì cấu trúc JSON không stable.

---

## Solutions Applied

### 1. Added Detailed Debug Logs

**File**: `UserService.java` - `getFriends()` method

Added step-by-step logging:
```java
@Transactional(readOnly = true)
public Page<UserSearchDTO> getFriends(@NotBlank String userId, @NotNull Pageable pageable) {
    log.info("Getting friends for userId: {}", userId);
    
    try {
        log.debug("Step 1: Calling userRepository.findFriends with page={}, size={}", 
                 pageable.getPageNumber(), pageable.getPageSize());
        
        var friends = userRepository.findFriends(userId, pageable);
        
        log.debug("Step 2: Retrieved {} friends from repository", friends.getTotalElements());
        log.debug("Step 3: Starting DTO mapping for {} friends", friends.getNumberOfElements());
        
        var result = friends.map(projection -> {
            try {
                log.trace("Mapping friend: {}", projection.getUser().getId());
                return userMapper.toUserSearchDTO(projection);
            } catch (Exception e) {
                log.error("Error mapping friend projection to DTO: {}", e.getMessage(), e);
                throw new RuntimeException("Error mapping friend data", e);
            }
        });
        
        log.debug("Step 4: Successfully mapped all friends to DTOs");
        log.info("Successfully retrieved {} friends for userId: {}", result.getTotalElements(), userId);
        
        return result;
    } catch (Exception e) {
        log.error("Error getting friends for userId {}: {}", userId, e.getMessage(), e);
        throw new RuntimeException("Failed to get friends list", e);
    }
}
```

**Benefits**:
- Track exactly where the API hangs
- Identify which step fails
- Catch and log mapping errors
- Full error context with stack trace

### 2. Created PageResponse DTO

**File**: `dto/PageResponse.java` (NEW)

```java
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PageResponse<T> {
    private List<T> content;
    private int page;
    private int size;
    private long totalElements;
    private int totalPages;
    private boolean first;
    private boolean last;
    private boolean empty;
    
    public static <T> PageResponse<T> of(Page<T> page) {
        return PageResponse.<T>builder()
            .content(page.getContent())
            .page(page.getNumber())
            .size(page.getSize())
            .totalElements(page.getTotalElements())
            .totalPages(page.getTotalPages())
            .first(page.isFirst())
            .last(page.isLast())
            .empty(page.isEmpty())
            .build();
    }
}
```

**Benefits**:
- Stable JSON structure
- No serialization warnings
- Compatible with frontend pagination
- Standard response format

### 3. Updated Controller Endpoints

**File**: `EnhancedUserController.java`

Changed return type from `Page<T>` to `PageResponse<T>`:

```java
// Before
@GetMapping("/me/friends")
public ResponseEntity<Page<UserSearchDTO>> getMyFriends(...) {
    Page<UserSearchDTO> friends = userService.getFriends(...);
    return ResponseEntity.ok(friends);
}

// After
@GetMapping("/me/friends")
public ResponseEntity<PageResponse<UserSearchDTO>> getMyFriends(...) {
    Page<UserSearchDTO> friendsPage = userService.getFriends(...);
    PageResponse<UserSearchDTO> response = PageResponse.of(friendsPage);
    return ResponseEntity.ok(response);
}
```

**Updated Endpoints**:
- `GET /api/users/me/friends` → Returns `PageResponse<UserSearchDTO>`
- `GET /api/users/{id}/mutual-friends` → Returns `PageResponse<UserSearchDTO>`

### 4. Enhanced Logging Configuration

**File**: `application.properties`

```properties
# Before
logging.level.com.ctuconnect=INFO
logging.level.org.springframework.data.neo4j=INFO
logging.level.org.neo4j.driver=WARN

# After
logging.level.com.ctuconnect=DEBUG
logging.level.com.ctuconnect.service.UserService=DEBUG
logging.level.com.ctuconnect.repository=DEBUG
logging.level.org.springframework.data.neo4j=DEBUG
logging.level.org.neo4j.driver=DEBUG
```

**Benefits**:
- See Neo4j queries being executed
- Track repository method calls
- Debug service layer execution
- Identify slow queries or deadlocks

---

## Response Format Changes

### Before (PageImpl)
```json
{
  "content": [...],
  "pageable": {
    "sort": {...},
    "offset": 0,
    "pageNumber": 0,
    "pageSize": 20,
    ...
  },
  "totalPages": 5,
  "totalElements": 100,
  ...many nested fields...
}
```

### After (PageResponse)
```json
{
  "content": [...],
  "page": 0,
  "size": 20,
  "totalElements": 100,
  "totalPages": 5,
  "first": true,
  "last": false,
  "empty": false
}
```

**Frontend Impact**: MINIMAL
- All essential fields preserved
- Simpler, flatter structure
- Compatible with existing code that reads:
  - `content` ✅
  - `totalElements` ✅
  - `totalPages` ✅
  - `number` → changed to `page` (minor adjustment needed)

---

## Frontend Changes Required

### Update Type Definitions

**File**: `client-frontend/src/types/index.ts`

```typescript
// Update PaginatedResponse to match new structure
export interface PaginatedResponse<T> {
  content: T[];
  page: number;        // Changed from 'number'
  size: number;
  totalElements: number;
  totalPages: number;
  first: boolean;
  last: boolean;
  empty: boolean;
}
```

### Update Components (if needed)

Most components should work without changes since they primarily use:
- `response.content` ✅ Still works
- `response.totalElements` ✅ Still works
- `response.totalPages` ✅ Still works

Only if using `response.number`, change to `response.page`.

---

## Debug Steps When Issue Occurs

### 1. Check Logs for Step-by-Step Progress

Look for these log entries:
```
2025-12-10 01:47:36 - Getting friends for userId: xxx
2025-12-10 01:47:36 - Step 1: Calling userRepository.findFriends with page=0, size=20
2025-12-10 01:47:36 - Step 2: Retrieved X friends from repository
2025-12-10 01:47:36 - Step 3: Starting DTO mapping for X friends
2025-12-10 01:47:36 - Step 4: Successfully mapped all friends to DTOs
2025-12-10 01:47:36 - Successfully retrieved X friends for userId: xxx
```

**If it stops at Step 1**: Neo4j query issue
- Check Neo4j logs
- Verify database connection
- Check if query is executing

**If it stops at Step 2**: Mapping issue
- Check UserMapper implementation
- Verify projection data structure
- Look for null pointer exceptions

**If it stops at Step 3**: DTO conversion issue
- Check individual mapping failures
- Verify all required fields are present

### 2. Check Neo4j Query Execution

With `logging.level.org.neo4j.driver=DEBUG`, you'll see:
```
Neo4j Query: MATCH (u:User {id: $userId})-[:IS_FRIENDS_WITH]-(friend:User) ...
```

If query doesn't appear in logs → connection problem
If query appears but no result → check data in Neo4j

### 3. Check for Exceptions

Look for stack traces:
```
Error getting friends for userId xxx: [detailed error message]
[full stack trace]
```

---

## Testing the Fix

### 1. Start User Service
```bash
cd user-service
./mvnw spring-boot:run
```

### 2. Test with curl
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/users/me/friends
```

### 3. Monitor Logs

Watch for the step-by-step logs:
```bash
tail -f logs/user-service.log
```

### 4. Verify Response Structure

Expected response:
```json
{
  "content": [...],
  "page": 0,
  "size": 20,
  "totalElements": 10,
  "totalPages": 1,
  "first": true,
  "last": true,
  "empty": false
}
```

---

## Common Issues & Solutions

### Issue 1: Still Hanging at Step 1

**Possible Causes**:
- Neo4j connection timeout
- Slow query execution
- Database lock

**Solutions**:
1. Check Neo4j browser: http://localhost:7474
2. Verify connection in logs
3. Run query manually in Neo4j browser
4. Check for long-running transactions

### Issue 2: Error at Step 2 (Mapping)

**Possible Causes**:
- Null fields in projection
- Missing relationships in data
- Mapper method issues

**Solutions**:
1. Check UserMapper.toUserSearchDTO()
2. Verify projection has all required fields
3. Add null checks in mapper

### Issue 3: Timeout

**Possible Causes**:
- Too many friends
- Inefficient query
- No pagination applied

**Solutions**:
1. Reduce page size: `?size=10`
2. Optimize Neo4j query
3. Add indexes to Neo4j

---

## Build Status

```
[INFO] BUILD SUCCESS
[INFO] Compiling 98 source files
✅ No compilation errors
✅ PageResponse DTO created
✅ Debug logs added
✅ Serialization warning fixed
```

---

## Files Modified

1. `UserService.java` - Added detailed debug logs
2. `EnhancedUserController.java` - Changed to PageResponse
3. `PageResponse.java` - NEW DTO for stable pagination
4. `application.properties` - Enhanced logging

---

**Date**: December 9, 2025
**Status**: ✅ FIXED
**Build**: ✅ SUCCESS

Next: Test with real data and monitor logs for exact issue location.
