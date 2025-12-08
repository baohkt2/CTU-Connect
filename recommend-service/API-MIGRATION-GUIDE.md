# API Migration Guide - Recommend Service

## Overview
This guide helps migrate from old endpoints to new unified endpoints after refactoring.

## Java Service Endpoints Migration

### ❌ OLD Endpoints (DEPRECATED)

```
# Old FeedController
GET  /api/recommendation/feed
POST /api/recommendation/interaction
POST /api/recommendation/cache/invalidate

# Old RecommendationController  
GET  /api/recommend/posts
POST /api/recommend/posts
POST /api/recommend/feedback
POST /api/recommend/embedding/rebuild
POST /api/recommend/rank/rebuild
DELETE /api/recommend/cache/{userId}
GET  /api/recommend/health
```

### ✅ NEW Endpoints (UNIFIED)

```
# Main Feed Endpoint
GET  /api/recommendations/feed?userId={id}&page=0&size=20

# Interaction Recording
POST /api/recommendations/interaction
Body: {
  "userId": "string",
  "postId": "string", 
  "type": "VIEW|LIKE|COMMENT|SHARE",
  "viewDuration": 0.0,
  "context": {}
}

# Cache Management
POST   /api/recommendations/refresh?userId={id}
DELETE /api/recommendations/cache/{userId}

# Health Checks
GET /api/recommendations/health
GET /api/recommendations/health/python
```

## Migration Examples

### Example 1: Get Personalized Feed

**OLD**:
```javascript
// Old way (multiple possible endpoints)
GET /api/recommendation/feed?userId=123&page=0&size=20
// OR
GET /api/recommend/posts?userId=123&page=0&size=20
```

**NEW**:
```javascript
// Unified way
GET /api/recommendations/feed?userId=123&page=0&size=20

// Response:
{
  "userId": "123",
  "recommendations": [
    {
      "postId": "p1",
      "authorId": "a1",
      "content": "...",
      "score": 0.95,
      "createdAt": "2024-12-08T10:00:00"
    }
  ],
  "totalCount": 20,
  "page": 0,
  "size": 20,
  "generatedAt": "2024-12-08T10:00:00",
  "processingTimeMs": 150
}
```

### Example 2: Record User Interaction

**OLD**:
```javascript
// Old way (FeedController)
POST /api/recommendation/interaction
{
  "userId": "123",
  "postId": "p1",
  "type": "like",
  "viewDuration": 5.2,
  "context": {}
}
```

**NEW**:
```javascript
// Same endpoint, unified path
POST /api/recommendations/interaction
{
  "userId": "123",
  "postId": "p1",
  "type": "LIKE",  // uppercase
  "viewDuration": 5.2,
  "context": {}
}

// Response:
{
  "status": "success",
  "message": "Interaction recorded"
}
```

### Example 3: Refresh User Cache

**OLD**:
```javascript
// Old way (FeedController)
POST /api/recommendation/cache/invalidate?userId=123
```

**NEW**:
```javascript
// New unified way
POST /api/recommendations/refresh?userId=123

// Response:
{
  "status": "success",
  "message": "Cache refreshed for user: 123"
}

// Alternative (RESTful way):
DELETE /api/recommendations/cache/123
```

### Example 4: Health Check

**OLD**:
```javascript
GET /api/recommend/health
GET /api/recommendation/health/python-service
```

**NEW**:
```javascript
// Java service health
GET /api/recommendations/health

// Python service health
GET /api/recommendations/health/python
```

## Frontend Migration Checklist

### Step 1: Update API Base URLs
```javascript
// OLD
const BASE_URL_OLD_1 = '/api/recommendation';
const BASE_URL_OLD_2 = '/api/recommend';

// NEW
const BASE_URL = '/api/recommendations';
```

### Step 2: Update Feed Fetching
```javascript
// OLD
async function getFeed(userId, page = 0, size = 20) {
  const response = await fetch(
    `/api/recommendation/feed?userId=${userId}&page=${page}&size=${size}`
  );
  return await response.json();
}

// NEW
async function getFeed(userId, page = 0, size = 20) {
  const response = await fetch(
    `/api/recommendations/feed?userId=${userId}&page=${page}&size=${size}`
  );
  return await response.json();
}
```

### Step 3: Update Interaction Recording
```javascript
// OLD
async function recordInteraction(userId, postId, type, duration) {
  await fetch('/api/recommendation/interaction', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      userId,
      postId,
      type: type.toLowerCase(),  // old: lowercase
      viewDuration: duration
    })
  });
}

// NEW
async function recordInteraction(userId, postId, type, duration) {
  await fetch('/api/recommendations/interaction', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      userId,
      postId,
      type: type.toUpperCase(),  // new: uppercase
      viewDuration: duration
    })
  });
}
```

### Step 4: Update Cache Refresh
```javascript
// OLD
async function refreshCache(userId) {
  await fetch(`/api/recommendation/cache/invalidate?userId=${userId}`, {
    method: 'POST'
  });
}

// NEW
async function refreshCache(userId) {
  await fetch(`/api/recommendations/refresh?userId=${userId}`, {
    method: 'POST'
  });
}
```

## Python Service (No Changes for External Clients)

Python service endpoints remain the same:
```
POST /embed/post
POST /embed/post/batch
POST /embed/user
POST /similarity
POST /similarity/batch
POST /api/model/predict
GET  /health
```

**Note**: These are internal endpoints, typically only called by Java service via `PythonModelServiceClient`.

## Backward Compatibility

Currently: **NO backward compatibility** for old endpoints.

If you need gradual migration:
1. Keep old controllers temporarily
2. Add deprecation warnings
3. Log warnings when old endpoints are hit
4. Remove after migration complete

## Testing After Migration

### 1. Test Feed Endpoint
```bash
curl -X GET "http://localhost:8081/api/recommendations/feed?userId=test123&size=10"
```

### 2. Test Interaction Recording
```bash
curl -X POST "http://localhost:8081/api/recommendations/interaction" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "test123",
    "postId": "post456",
    "type": "LIKE",
    "viewDuration": 5.5
  }'
```

### 3. Test Cache Refresh
```bash
curl -X POST "http://localhost:8081/api/recommendations/refresh?userId=test123"
```

### 4. Test Health
```bash
curl -X GET "http://localhost:8081/api/recommendations/health"
```

## Common Issues & Solutions

### Issue 1: 404 Not Found
**Problem**: Old endpoint path used
**Solution**: Update to new unified path `/api/recommendations/*`

### Issue 2: 400 Bad Request on interaction type
**Problem**: Using lowercase interaction type
**Solution**: Use uppercase: `LIKE`, `COMMENT`, `SHARE`, not `like`, `comment`, `share`

### Issue 3: Service not responding
**Problem**: Java service can't reach Python service
**Solution**: Verify Python service is running on port 8000
```bash
curl http://localhost:8000/health
```

## Support

If you encounter issues during migration:
1. Check logs: `recommend-service/java-api/logs/`
2. Check Python logs: `recommend-service/python-model/logs/`
3. Verify service health endpoints
4. Check network connectivity between services

## Timeline

- **2024-12-08**: New unified endpoints available
- **Future**: Old endpoints to be fully removed
- **Recommended**: Migrate immediately to avoid issues

---

**Questions?** Check ARCHITECTURE.md and OPTIMIZATION-SUMMARY.md for more details.
