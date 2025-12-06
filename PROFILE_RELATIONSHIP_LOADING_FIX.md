# Profile Relationship Loading Fix

## Problem
After updating profile with major/batch/gender, the `checkMyInfo` endpoint returned `false` even though the relationships exist in Neo4j database.

### Logs showing the issue:
```
2025-12-07 00:47:48 - Profile completion check: false
2025-12-07 00:47:48 - Profile details - fullName: true, studentId: true, major: false, batch: false, gender: false
```

But in Neo4j database, the relationships actually exist!

## Root Cause

**Spring Data Neo4j doesn't automatically load relationships when using custom `@Query` methods.**

### The Problem Code:

```java
// UserRepository.java
@Query("""
    MATCH (u:User {id: $userId})
    OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
    OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
    OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
    WITH u, m, b, g
    OPTIONAL MATCH (m)-[:HAS_MAJOR]-(f:Faculty)
    WITH u, m, b, g, f
    OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
    RETURN u, m, f, c, b, g
    LIMIT 1
    """)
Optional<UserEntity> findUserWithRelationships(@Param("userId") String userId);
```

**Issue**: Even though we MATCH the relationships, Spring Data Neo4j doesn't know how to map `m`, `b`, `g` back to the UserEntity's `major`, `batch`, `gender` fields.

### Why it failed:
1. Custom `@Query` returns raw Cypher results
2. Spring Data Neo4j expects specific return patterns to hydrate relationships
3. Returning multiple nodes `(u, m, f, c, b, g)` confuses the mapping
4. The UserEntity ends up with `major=null`, `batch=null`, `gender=null`

## Solution

**Use `findById()` instead of custom query!**

Spring Data Neo4j's built-in `findById()` method:
- Automatically loads all `@Relationship` annotated fields
- Uses proper depth and mapping
- Handles all relationship types correctly

### Fixed Code:

```java
// UserService.java
@Transactional(readOnly = true)
public UserProfileDTO getUserProfile(@NotBlank String userId) {
    log.info("Fetching user profile for userId: {}", userId);

    // Use standard findById which loads relationships automatically
    var user = userRepository.findById(userId)
        .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));
    
    // Log for debugging
    log.info("User relationships loaded - major: {}, batch: {}, gender: {}", 
            user.getMajor() != null ? user.getMajor().getName() : "null", 
            user.getBatch() != null ? user.getBatch().getYear() : "null", 
            user.getGender() != null ? user.getGender().getName() : "null");

    return userMapper.toUserProfileDTO(user);
}
```

### Simplified Repository Query:

```java
// UserRepository.java
@Query("""
    MATCH (u:User {id: $userId})
    RETURN u
    """)
Optional<UserEntity> findUserWithRelationships(@Param("userId") String userId);
```

Or better yet, just remove it and use `findById()`!

## How Spring Data Neo4j Works

### With Custom @Query:
```
Custom Query → Raw Cypher Results → Manual Mapping Required
                                      ↓
                              Often relationships = null
```

### With findById():
```
findById() → SDN's Smart Loading → Auto-loads @Relationship fields
                                    ↓
                              Full entity with relationships!
```

## Testing

### Before Fix:
```bash
curl GET /api/users/checkMyInfo
# Response: false

# Logs:
# major: false, batch: false, gender: false
```

### After Fix:
```bash
curl GET /api/users/checkMyInfo
# Response: true

# Logs:
# major: Công nghệ thông tin, batch: 2021, gender: Male
```

## Files Modified

1. **UserService.java** - `getUserProfile()`
   - Changed from `findUserWithRelationships()` to `findById()`
   - Added debug logging

2. **UserRepository.java** - `findUserWithRelationships()`
   - Simplified query to just `RETURN u`
   - Or can be removed entirely

## Why This Works

1. **`findById()` is Spring Data Neo4j's native method**
   - Knows how to load `@Relationship` fields
   - Uses proper mapping strategy
   - Loads with appropriate depth

2. **UserEntity has proper annotations**:
   ```java
   @Relationship(type = "ENROLLED_IN", direction = Relationship.Direction.OUTGOING)
   private MajorEntity major;
   
   @Relationship(type = "IN_BATCH", direction = Relationship.Direction.OUTGOING)
   private BatchEntity batch;
   
   @Relationship(type = "HAS_GENDER", direction = Relationship.Direction.OUTGOING)
   private GenderEntity gender;
   ```

3. **SDN knows to fetch these automatically**

## Key Learnings

### ✅ DO:
- Use `findById()` for loading entities with relationships
- Let Spring Data Neo4j handle relationship loading
- Trust the framework's built-in methods

### ❌ DON'T:
- Write custom `@Query` for simple entity loading
- Try to manually map relationships from Cypher results
- Return multiple nodes when you just need the entity

## Verification Query

Run this in Neo4j Browser to verify relationships exist:

```cypher
MATCH (u:User {id: '31ba8a23-8a4e-4b24-99c2-0d768e617e71'})
OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
RETURN u.fullName as fullName,
       u.studentId as studentId,
       m.name as major,
       b.year as batch,
       g.name as gender
```

Should return all fields populated.

## Expected Behavior Now

1. User updates profile → Relationships created via custom Cypher
2. System fetches user profile → Uses `findById()`
3. Spring Data Neo4j loads all relationships automatically
4. Profile completion check → All fields present → Returns `true`
5. User not redirected to update page anymore! ✅

## Notes

- The custom update queries (`updateUserMajor`, etc.) still work fine
- They create/update relationships in Neo4j correctly
- The issue was only with **reading** the relationships
- Using `findById()` for reading solves it completely
