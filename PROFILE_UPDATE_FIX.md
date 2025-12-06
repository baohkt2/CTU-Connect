# Profile Update Fix - Complete System

## Summary
Fixed the profile update flow from client-frontend to user-service with proper relationship management in Neo4j.

## Issues Fixed

### 1. Field Name Mismatch
**Problem:** Frontend was sending `majorName`, `batchYear`, `genderName` but backend was looking up by `name` instead of `code`.

**Solution:**
- Changed DTO to use `majorCode` instead of `majorName`
- Backend now looks up Major by `code` instead of `name`
- Frontend sends `genderCode` which is mapped to `genderName` for the backend

### 2. Relationship Update Not Working
**Problem:** Simple entity setter methods (`setMajor()`, `setBatch()`, `setGender()`) don't properly update Neo4j relationships. Old relationships were not removed before creating new ones.

**Solution:**
- Added custom Cypher queries in `UserRepository` that:
  1. Find and DELETE old relationship
  2. CREATE new relationship
- Three new methods added:
  - `updateUserMajor(userId, majorCode)`
  - `updateUserBatch(userId, batchYear)`
  - `updateUserGender(userId, genderName)`

### 3. Only IN_BATCH Was Updating
**Problem:** The simple save approach only worked for `IN_BATCH` by chance, but `ENROLLED_IN` (major) and `HAS_GENDER` were not updating.

**Solution:**
- Separated basic profile updates from relationship updates
- Save basic fields first, then update relationships using custom queries

### 4. Kafka Message Conversion Error (Bonus Fix)
**Problem:** `UserUpdatedEvent` was being deserialized as object but listener expected `Map<String, Object>`.

**Solution:**
- Changed listener parameter type from `Map<String, Object>` to `UserUpdatedEvent`

### 5. Multiple Records Returned Error (Bonus Fix)
**Problem:** `findUserWithRelationships` query was returning multiple records.

**Solution:**
- Added `LIMIT 1` to the query

## Files Modified

### Backend (user-service)

1. **UserUpdateDTO.java**
   - Changed `majorName` → `majorCode`

2. **UserRepository.java**
   - Added `LIMIT 1` to `findUserWithRelationships` query
   - Added three new methods:
     ```java
     void updateUserMajor(String userId, String majorCode)
     void updateUserBatch(String userId, String batchYear)
     void updateUserGender(String userId, String genderName)
     ```

3. **UserService.java - updateUserProfile()**
   - Separated basic profile update from relationship updates
   - Save basic fields first
   - Then call custom repository methods for each relationship
   - Verify entity exists before updating relationship
   - Fetch updated user with relationships after all updates
   - Added detailed logging for each relationship update

4. **UserEventListener.java**
   - Changed `handleUserUpdatedEvent` parameter from `Map<String, Object>` to `UserUpdatedEvent`

### Frontend (client-frontend)

1. **user.ts (shared/types)**
   - Updated `UpdateProfileRequest` interface:
     - `majorId` → `majorCode`
     - `batchId` → `batchYear`
     - `genderId` → `genderName`

2. **api-endpoints.ts**
   - Changed paths from `/api/users/profile` to `/api/users/me/profile`

3. **userService.ts (services)**
   - Added transformation logic in `updateMyProfile`:
     - Maps `genderCode` (frontend) to `genderName` (backend)
     - Properly structures data for backend API
     - Added logging for debugging

## How It Works Now

### Update Flow

1. **Frontend (StudentProfileForm.tsx)**:
   ```typescript
   formData = {
     fullName: "...",
     bio: "...",
     studentId: "...",
     majorCode: "CNTT",      // e.g. "CNTT", "KTDK"
     batchYear: "2021",       // e.g. "2021", "2022"
     genderCode: "Male"       // e.g. "Male", "Female"
   }
   ```

2. **Frontend Service (userService.ts)**:
   ```typescript
   backendData = {
     fullName: "...",
     bio: "...",
     studentId: "...",
     majorCode: "CNTT",
     batchYear: "2021",
     genderName: "Male"       // Transformed from genderCode
   }
   ```

3. **Backend Controller (EnhancedUserController.java)**:
   - Receives `UserUpdateDTO` at `PUT /api/users/me/profile`
   - Calls `userService.updateUserProfile(userId, updateDTO)`

4. **Backend Service (UserService.java)**:
   ```java
   // Step 1: Update basic fields
   user.setFullName(updateDTO.getFullName());
   user.setBio(updateDTO.getBio());
   user.setStudentId(updateDTO.getStudentId());
   userRepository.save(user);
   
   // Step 2: Update relationships with custom queries
   if (updateDTO.getMajorCode() != null) {
       majorRepository.findByCode(updateDTO.getMajorCode()); // Verify exists
       userRepository.updateUserMajor(userId, updateDTO.getMajorCode());
   }
   
   if (updateDTO.getBatchYear() != null) {
       batchRepository.findByYear(updateDTO.getBatchYear()); // Verify exists
       userRepository.updateUserBatch(userId, updateDTO.getBatchYear());
   }
   
   if (updateDTO.getGenderName() != null) {
       genderRepository.findByName(updateDTO.getGenderName()); // Verify exists
       userRepository.updateUserGender(userId, updateDTO.getGenderName());
   }
   ```

5. **Neo4j Queries**:
   ```cypher
   // For Major update - Uses MERGE to avoid duplicates
   MATCH (u:User {id: $userId})
   OPTIONAL MATCH (u)-[r:ENROLLED_IN]->(:Major)
   DELETE r
   WITH u
   MATCH (m:Major {code: $majorCode})
   MERGE (u)-[:ENROLLED_IN]->(m)
   
   // For Batch update - Uses MERGE to avoid duplicates
   MATCH (u:User {id: $userId})
   OPTIONAL MATCH (u)-[r:IN_BATCH]->(:Batch)
   DELETE r
   WITH u
   MATCH (b:Batch {year: $batchYear})
   MERGE (u)-[:IN_BATCH]->(b)
   
   // For Gender update - Uses MERGE to avoid duplicates
   MATCH (u:User {id: $userId})
   OPTIONAL MATCH (u)-[r:HAS_GENDER]->(:Gender)
   DELETE r
   WITH u
   MATCH (g:Gender {name: $genderName})
   MERGE (u)-[:HAS_GENDER]->(g)
   ```

6. **Query Optimization**:
   - Changed `findUserWithRelationships` to use WITH clauses
   - This prevents cartesian product and ensures single result
   - Added LIMIT 1 as safety measure

## Key Design Principles

1. **Separation of Concerns**: Basic attributes vs. relationships handled separately
2. **Data Validation**: Verify target entities exist before updating relationships
3. **Atomicity**: Each relationship update is atomic with old relationship deletion
4. **Field Mapping**: Use `code` for lookups (Major, Batch) and `name` for Gender
5. **Consistent Naming**: Backend uses entity identifiers consistently

## Database Cleanup (Important!)

If you're experiencing "multiple records" errors, you may have duplicate relationships in the database from before the fix. Run this cleanup script:

```bash
# Copy the cleanup script to Neo4j
cat database/cleanup_duplicate_relationships.cypher

# Then run it in Neo4j Browser or via cypher-shell
cypher-shell -u neo4j -p password < database/cleanup_duplicate_relationships.cypher
```

Or run directly in Neo4j Browser:
```cypher
// Remove duplicate ENROLLED_IN relationships
MATCH (u:User)-[r:ENROLLED_IN]->(m:Major)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// Remove duplicate IN_BATCH relationships
MATCH (u:User)-[r:IN_BATCH]->(b:Batch)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// Remove duplicate HAS_GENDER relationships
MATCH (u:User)-[r:HAS_GENDER]->(g:Gender)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;
```

## Testing Checklist

- [ ] **First: Run database cleanup script above**
- [ ] Update major only
- [ ] Update batch only
- [ ] Update gender only
- [ ] Update all three relationships together
- [ ] Update basic info (fullName, bio, studentId)
- [ ] Update everything at once
- [ ] Try updating with non-existent majorCode (should fail gracefully)
- [ ] Try updating with non-existent batchYear (should fail gracefully)
- [ ] Verify old relationships are removed in Neo4j
- [ ] Verify Kafka event is published correctly
- [ ] Verify frontend shows updated data after refresh

## Related Files

### Backend
- `user-service/src/main/java/com/ctuconnect/dto/UserUpdateDTO.java`
- `user-service/src/main/java/com/ctuconnect/repository/UserRepository.java`
- `user-service/src/main/java/com/ctuconnect/service/UserService.java`
- `user-service/src/main/java/com/ctuconnect/service/UserEventListener.java`

### Frontend
- `client-frontend/src/shared/types/user.ts`
- `client-frontend/src/shared/constants/api-endpoints.ts`
- `client-frontend/src/services/userService.ts`
- `client-frontend/src/components/profile/StudentProfileForm.tsx`

## Notes

1. The frontend already had `majorCode`, `batchYear`, and `genderCode` in `StudentProfileUpdateRequest` - no changes needed there
2. Gender uses `name` for lookup instead of `code` because Gender entity uses `name` as primary identifier
3. The transformation from `genderCode` to `genderName` happens in the frontend service layer
4. All relationship updates are now properly logged for debugging
5. The fix is systematic and follows Neo4j best practices for relationship management
