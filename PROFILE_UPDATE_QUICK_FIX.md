# Profile Update - Quick Fix Summary

## The Problem
Getting error: "Expected a result with a single record, but this result contains at least one more"

## Root Cause
1. Duplicate relationships already exist in database (from before the fix)
2. Query cartesian product when fetching user with relationships

## Quick Solution

### Step 1: Clean Database (REQUIRED)
Open Neo4j Browser (http://localhost:7474) and run:

```cypher
// Remove duplicate ENROLLED_IN
MATCH (u:User)-[r:ENROLLED_IN]->(m:Major)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// Remove duplicate IN_BATCH
MATCH (u:User)-[r:IN_BATCH]->(b:Batch)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// Remove duplicate HAS_GENDER
MATCH (u:User)-[r:HAS_GENDER]->(g:Gender)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;
```

### Step 2: Restart Services
```bash
# Restart user-service to pick up the fixes
# The code has been updated to:
# 1. Use MERGE instead of CREATE (prevents duplicates)
# 2. Better query structure with WITH clauses
# 3. LIMIT 1 safety
```

### Step 3: Test Profile Update
Now try updating profile again through the frontend or API.

## What Was Fixed

### Backend Changes
1. **UserRepository.java**:
   - Changed `CREATE` to `MERGE` in relationship update queries
   - Removed `RETURN u` from void methods
   - Added WITH clauses in `findUserWithRelationships` to prevent cartesian product

2. **Query Structure**:
   ```cypher
   // OLD (could create duplicates)
   CREATE (u)-[:ENROLLED_IN]->(m)
   
   // NEW (prevents duplicates)
   MERGE (u)-[:ENROLLED_IN]->(m)
   ```

3. **Fetch Query**:
   ```cypher
   // OLD (could cause cartesian product)
   MATCH (u:User {id: $userId})
   OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
   OPTIONAL MATCH (m)-[:HAS_MAJOR]-(f:Faculty)
   OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
   ...
   
   // NEW (uses WITH to control cardinality)
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
   ```

## Verification

After cleanup, verify no duplicates exist:

```cypher
MATCH (u:User)
OPTIONAL MATCH (u)-[r1:ENROLLED_IN]->(:Major)
WITH u, COUNT(r1) as majorCount
OPTIONAL MATCH (u)-[r2:IN_BATCH]->(:Batch)
WITH u, majorCount, COUNT(r2) as batchCount
OPTIONAL MATCH (u)-[r3:HAS_GENDER]->(:Gender)
RETURN u.email, majorCount, batchCount, COUNT(r3) as genderCount
ORDER BY u.email;
```

Each count should be 0 or 1 (never > 1).

## Expected Behavior After Fix

1. ✅ No more "multiple records" error
2. ✅ Profile updates successfully
3. ✅ Each user has at most ONE of each relationship type
4. ✅ Old relationships are properly removed before adding new ones
5. ✅ Kafka events publish correctly

## If Still Having Issues

1. Check if database cleanup was run successfully
2. Verify all instances of user-service are restarted
3. Check Neo4j logs for any constraint violations
4. Run the verification query above
5. Check user-service logs for detailed error messages

## Files Modified
- `user-service/src/main/java/com/ctuconnect/repository/UserRepository.java`
- Created `database/cleanup_duplicate_relationships.cypher`
- Updated `PROFILE_UPDATE_FIX.md`
- Updated `PROFILE_UPDATE_TEST_GUIDE.md`
