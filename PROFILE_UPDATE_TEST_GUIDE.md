# Profile Update Testing Guide

## IMPORTANT: Database Cleanup First!

Before testing, clean up any duplicate relationships in Neo4j:

```cypher
// Run this in Neo4j Browser (http://localhost:7474)

// 1. Remove duplicate ENROLLED_IN relationships
MATCH (u:User)-[r:ENROLLED_IN]->(m:Major)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// 2. Remove duplicate IN_BATCH relationships
MATCH (u:User)-[r:IN_BATCH]->(b:Batch)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// 3. Remove duplicate HAS_GENDER relationships
MATCH (u:User)-[r:HAS_GENDER]->(g:Gender)
WITH u, COLLECT(r) as rels
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// 4. Verify cleanup (should return no results)
MATCH (u:User)
OPTIONAL MATCH (u)-[r1:ENROLLED_IN]->(:Major)
WITH u, COUNT(r1) as majorCount
OPTIONAL MATCH (u)-[r2:IN_BATCH]->(:Batch)
WITH u, majorCount, COUNT(r2) as batchCount
OPTIONAL MATCH (u)-[r3:HAS_GENDER]->(:Gender)
WITH u, majorCount, batchCount, COUNT(r3) as genderCount
WHERE majorCount > 1 OR batchCount > 1 OR genderCount > 1
RETURN u.id, u.email, majorCount, batchCount, genderCount;
```

## Quick Test with cURL

### 1. Get Authentication Token
```bash
# Login first to get token
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "baob2110069@student.ctu.edu.vn",
    "password": "your_password"
  }'

# Save the token from response
export TOKEN="your_jwt_token_here"
```

### 2. Test Profile Update

#### Update All Fields
```bash
curl -X PUT http://localhost:8080/api/users/me/profile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "fullName": "Nguyen Van A",
    "bio": "CTU Student",
    "studentId": "B2110069",
    "majorCode": "CNTT",
    "batchYear": "2021",
    "genderName": "Male"
  }'
```

#### Update Only Major
```bash
curl -X PUT http://localhost:8080/api/users/me/profile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "majorCode": "KTDK"
  }'
```

#### Update Only Batch
```bash
curl -X PUT http://localhost:8080/api/users/me/profile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "batchYear": "2022"
  }'
```

#### Update Only Gender
```bash
curl -X PUT http://localhost:8080/api/users/me/profile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "genderName": "Female"
  }'
```

### 3. Verify in Neo4j

```cypher
// Check user's relationships
MATCH (u:User {id: 'your-user-id'})
OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
RETURN u.fullName, u.studentId, m.code as major, b.year as batch, g.name as gender
```

```cypher
// Check if old relationships are removed
MATCH (u:User {id: 'your-user-id'})-[r]->()
RETURN type(r) as relationship, count(r) as count
// Should show only 1 relationship of each type (ENROLLED_IN, IN_BATCH, HAS_GENDER)
```

## Frontend Testing

### 1. Open Profile Update Page
Navigate to: `http://localhost:3000/profile/update`

### 2. Fill in the Form
- **Full Name**: Enter your full name
- **Student ID**: Enter student ID (e.g., B2110069)
- **Bio**: Optional description
- **College**: Select from dropdown
- **Faculty**: Select from dropdown (filtered by college)
- **Major**: Select from dropdown (filtered by faculty)
- **Batch**: Select year (e.g., 2021)
- **Gender**: Select from dropdown

### 3. Submit and Verify
- Click "Update Profile" button
- Should see success toast message
- Check browser console for request/response
- Refresh page and verify data is saved

## Common Issues and Solutions

### Issue 1: "Major not found with code: XXX"
**Cause:** The majorCode doesn't exist in database
**Solution:** Check available major codes in Neo4j:
```cypher
MATCH (m:Major) RETURN m.code, m.name ORDER BY m.code
```

### Issue 2: "Batch not found: YYYY"
**Cause:** The batch year doesn't exist in database
**Solution:** Check available batches:
```cypher
MATCH (b:Batch) RETURN b.year ORDER BY b.year
```

### Issue 3: "Gender not found: XXX"
**Cause:** The gender name doesn't exist in database
**Solution:** Check available genders:
```cypher
MATCH (g:Gender) RETURN g.name
```

### Issue 4: Relationship not updating
**Cause:** Old custom code might be interfering
**Solution:** 
1. Check logs in user-service console
2. Look for messages like: "Updated major relationship for userId: XXX"
3. Verify Neo4j query execution in logs

### Issue 5: Frontend sends wrong field names
**Cause:** Caching or old code
**Solution:**
1. Clear browser cache
2. Restart frontend dev server
3. Check browser Network tab to see actual request payload

## Expected Behaviors

### ✅ Correct Behavior
1. Old relationship is deleted before creating new one
2. Only ONE relationship of each type exists (ENROLLED_IN, IN_BATCH, HAS_GENDER)
3. User info is updated immediately
4. Kafka event is published
5. Success message appears in frontend

### ❌ Incorrect Behavior (Fixed)
1. Multiple ENROLLED_IN relationships to different majors
2. Relationships not updating at all
3. 404 Not Found error
4. Field name mismatch errors

## Monitoring Logs

### Backend Logs to Watch
```
# Successful update
2025-12-06 23:45:00 - Updating user profile for userId: 31ba8a23-8a4e-4b24-99c2-0d768e617e71
2025-12-06 23:45:00 - Updated major relationship for userId: 31ba8a23-8a4e-4b24-99c2-0d768e617e71 to majorCode: CNTT
2025-12-06 23:45:00 - Updated batch relationship for userId: 31ba8a23-8a4e-4b24-99c2-0d768e617e71 to batchYear: 2021
2025-12-06 23:45:00 - Updated gender relationship for userId: 31ba8a23-8a4e-4b24-99c2-0d768e617e71 to genderName: Male
2025-12-06 23:45:00 - User profile updated successfully for userId: 31ba8a23-8a4e-4b24-99c2-0d768e617e71
2025-12-06 23:45:00 - Published user updated event for userId: 31ba8a23-8a4e-4b24-99c2-0d768e617e71
```

### Kafka Logs to Watch
```
2025-12-06 23:45:01 - Received user-updated event from topic 'user-updated': UserUpdatedEvent(userId=..., majorCode=CNTT, ...)
```

## Data Validation Queries

```cypher
// Count relationships per user
MATCH (u:User {id: 'your-user-id'})
OPTIONAL MATCH (u)-[r:ENROLLED_IN]->(:Major)
WITH u, count(r) as majorCount
OPTIONAL MATCH (u)-[r2:IN_BATCH]->(:Batch)
WITH u, majorCount, count(r2) as batchCount
OPTIONAL MATCH (u)-[r3:HAS_GENDER]->(:Gender)
RETURN majorCount, batchCount, count(r3) as genderCount
// Should all be 1 or 0

// Get full user profile
MATCH (u:User {id: 'your-user-id'})
OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)-[:HAS_MAJOR]-(f:Faculty)-[:HAS_FACULTY]-(c:College)
OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
RETURN u, m, f, c, b, g
```
