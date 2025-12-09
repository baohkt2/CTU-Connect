# Restart and Test Guide - Recommendation System Fixes

**Date:** 2025-12-09  
**Purpose:** Deploy and test Kafka user action and ML prediction fixes

## Prerequisites

- All fixes have been applied to code
- Java API rebuilt successfully (`mvn clean package`)
- Docker and docker-compose available
- Access to database for verification

---

## Step 1: Restart Recommend-Service

### Option A: Using docker-compose (Recommended)

```bash
# Navigate to project root
cd d:\LVTN\CTU-Connect-demo

# Rebuild and restart only recommend services
docker-compose up -d --build recommend-service
docker-compose up -d --build recommend-python

# Verify containers are running
docker ps | findstr recommend

# You should see:
# - recommend-service (Java API on port 8095)
# - recommend-python (Python ML on port 8000)
# - ctu-recommend-postgres (port 5435)
# - ctu-recommend-redis (port 6380)
```

### Option B: Start specific services

```bash
# If services are defined differently in your docker-compose.yml
docker-compose up -d --build java-api
docker-compose up -d --build python-model

# Or start all services
docker-compose up -d
```

---

## Step 2: Monitor Logs in Real-Time

Open **3 terminal windows** to monitor logs:

### Terminal 1: Recommend-Service Java API
```bash
docker logs -f ctu-connect-recommend-service 2>&1 | Select-String -Pattern "user_action|Received|Error|SUCCESS"
```

### Terminal 2: Recommend-Service Python Model
```bash
docker logs -f ctu-connect-recommend-python 2>&1 | Select-String -Pattern "Prediction|ERROR|scores"
```

### Terminal 3: Post-Service (to see event publishing)
```bash
docker logs -f post-service 2>&1 | Select-String -Pattern "Published|user_action|interaction"
```

---

## Step 3: Test User Interaction Flow

### 3.1 Like a Post

**From Frontend or Postman:**
```http
POST http://localhost:8080/api/posts/{postId}/interactions
Content-Type: application/json
Authorization: Bearer {your-token}

{
  "reaction": "LIKE"
}
```

**Expected Logs:**

**Post-Service:**
```
📤 Published user_action event: LIKE by user {userId} on post {postId}
```

**Recommend-Service Java:**
```
📥 Received user_action: LIKE by user {userId} on post {postId}
💾 Saved user feedback: {userId} -> {postId} (type: LIKE, value: 1.0)
📊 Updated engagement for post {postId}: likes=N, comments=M, shares=K
🗑️  Invalidated cache for user: {userId}
✅ Successfully processed user_action event: LIKE (feedback: LIKE)
```

**Verify in Database:**
```sql
-- Connect to recommend-postgres
docker exec -it ctu-recommend-postgres psql -U postgres -d recommendation_db

-- Check user feedback
SELECT * FROM user_feedback 
WHERE user_id = '{userId}' 
ORDER BY created_at DESC LIMIT 5;

-- Check post engagement
SELECT post_id, like_count, comment_count, share_count, popularity_score, updated_at
FROM post_embeddings 
WHERE post_id = '{postId}';

-- Should see like_count incremented
```

### 3.2 Comment on a Post

**From Frontend or Postman:**
```http
POST http://localhost:8080/api/posts/{postId}/comments
Content-Type: application/json
Authorization: Bearer {your-token}

{
  "content": "Great post!"
}
```

**Expected Logs:**

**Post-Service:**
```
📤 Published COMMENT user_action event for user {userId} on post {postId}
```

**Recommend-Service:**
```
📥 Received user_action: COMMENT by user {userId} on post {postId}
💾 Saved user feedback: {userId} -> {postId} (type: COMMENT, value: 2.0)
📊 Updated engagement for post {postId}: comments=N+1
```

**Verify in Database:**
```sql
-- Check comment feedback
SELECT feedback_type, feedback_value, created_at 
FROM user_feedback 
WHERE user_id = '{userId}' AND post_id = '{postId}' AND feedback_type = 'COMMENT';

-- Should return a row with feedback_value = 2.0
```

---

## Step 4: Test Recommendation Generation

### 4.1 Request Feed

**From Frontend or Postman:**
```http
GET http://localhost:8095/api/recommendations/feed?userId={userId}&page=0&size=10
Authorization: Bearer {your-token}
```

**Expected Logs:**

**Recommend-Service Java:**
```
========================================
📡 API REQUEST: GET /api/recommendations/feed
   User ID: {userId}
   Page: 0, Size: 10
========================================
🔄 Calling hybrid recommendation service for feed generation
```

**Recommend-Service Python:**
```
Prediction request for user: {userId}, candidates: N
Post abc123 scores: content_sim=0.4523, implicit_fb=0.6789, academic=0.2100, popularity=0.1234
Post def456 scores: content_sim=0.2891, implicit_fb=0.5012, academic=0.7654, popularity=0.4321
Post ghi789 scores: content_sim=0.6123, implicit_fb=0.3456, academic=0.4567, popularity=0.2890
Prediction completed: N posts ranked in 234.56ms
```

**Recommend-Service Java:**
```
========================================
📊 API RESPONSE: GET /api/recommendations/feed
   Total Items: N
   User ID: {userId}
📊 RECOMMENDED POSTS LIST:
   Format: [Rank] PostID -> Score
   ----------------------------------------
   [ 1] abc123 -> score: 0.5234
   [ 2] def456 -> score: 0.4987
   [ 3] ghi789 -> score: 0.4512
   ----------------------------------------
📈 SCORE STATISTICS:
   Max Score: 0.5234
   Min Score: 0.4512
   Avg Score: 0.4911
========================================
```

**Key Verification:**
- ✅ Scores are **different** for different posts (not all 0.3000)
- ✅ Python logs show **detailed scores** for each component
- ✅ No "NoneType multiplication" errors
- ✅ Response completes in < 500ms

### 4.2 Verify Score Diversity

**Check Response JSON:**
```json
{
  "posts": [
    {
      "postId": "abc123",
      "score": 0.5234,
      "contentSimilarity": 0.4523,
      "implicitFeedback": 0.6789,
      "academicScore": 0.2100,
      "popularityScore": 0.1234
    },
    {
      "postId": "def456",
      "score": 0.4987,
      "contentSimilarity": 0.2891,
      "implicitFeedback": 0.5012,
      "academicScore": 0.7654,
      "popularityScore": 0.4321
    }
  ]
}
```

**Validation:**
- ✅ Each post has **different score** values
- ✅ Score components (contentSimilarity, implicitFeedback, etc.) are present
- ✅ Scores are in valid range [0.0, 1.0]
- ✅ Posts are sorted by score descending

---

## Step 5: Test Post Creation & Embedding

### 5.1 Create New Post

**From Frontend or Postman:**
```http
POST http://localhost:8080/api/posts
Content-Type: application/json
Authorization: Bearer {your-token}

{
  "content": "Câu hỏi về tuyển sinh đại học CTU năm 2025",
  "category": "ADMISSION",
  "privacy": "PUBLIC"
}
```

**Expected Logs:**

**Post-Service:**
```
📤 POST_CREATED event published: {postId}
```

**Recommend-Service Java:**
```
📥 Received post event: POST_CREATED for {postId}
🤖 Generating embedding for post: {postId}
```

**Recommend-Service Python:**
```
Generating embedding for text: Câu hỏi về tuyển sinh đại học CTU năm 2025
✅ Embedding generated successfully: dimension=768
```

**Recommend-Service Java:**
```
✅ Embedding saved successfully for post: {postId}
```

**Verify in Database:**
```sql
SELECT post_id, score, like_count, comment_count, created_at,
       LENGTH(embedding::text) as embedding_size
FROM post_embeddings 
WHERE post_id = '{postId}';

-- Verify:
-- 1. Row exists
-- 2. score > 0 (not 0.0)
-- 3. embedding is not null
-- 4. embedding_size > 100 (should be ~768 dimensions)
```

---

## Step 6: Comprehensive Verification

### 6.1 Database Checks

```sql
-- Connect to recommend-postgres
docker exec -it ctu-recommend-postgres psql -U postgres -d recommendation_db

-- 1. Check user_feedback has data
SELECT COUNT(*) as total_feedback FROM user_feedback;
-- Should be > 0 after interactions

-- 2. Check feedback types distribution
SELECT feedback_type, COUNT(*) as count, AVG(feedback_value) as avg_value
FROM user_feedback
GROUP BY feedback_type
ORDER BY count DESC;

-- 3. Check post_embeddings have proper scores
SELECT 
    COUNT(*) as total_posts,
    AVG(score) as avg_score,
    MAX(score) as max_score,
    MIN(score) as min_score,
    AVG(like_count) as avg_likes,
    AVG(comment_count) as avg_comments
FROM post_embeddings;

-- 4. Check most engaged posts
SELECT post_id, like_count, comment_count, share_count, popularity_score
FROM post_embeddings
ORDER BY popularity_score DESC
LIMIT 10;

-- 5. Check cache in Redis
\q  # Exit psql

# Connect to Redis
docker exec -it ctu-recommend-redis redis-cli

# Check cached recommendations
KEYS recommendations:*
GET "recommendations:{userId}"

# Check cache invalidation working
# After a like, the key should be deleted or updated
```

### 6.2 API Response Validation

**Test with curl:**
```bash
# Get recommendations
curl -X GET "http://localhost:8095/api/recommendations/feed?userId={userId}&page=0&size=10" \
  -H "Authorization: Bearer {token}" | jq

# Validate response structure
# Should have:
# - "posts" array with N items
# - Each post has "postId", "score", "contentSimilarity", etc.
# - Scores are diverse (not all same value)
```

---

## Troubleshooting

### Issue 1: No user_action events received

**Symptoms:**
- Post-service logs show "Published user_action event"
- Recommend-service logs show nothing
- user_feedback table empty

**Check:**
```bash
# 1. Verify Kafka topic exists
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list | findstr user_action

# 2. Check messages in topic
docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic user_action --from-beginning --max-messages 5

# 3. Check recommend-service Kafka connection
docker logs ctu-connect-recommend-service 2>&1 | Select-String -Pattern "Kafka|consumer"

# 4. Restart recommend-service
docker-compose restart recommend-service
```

### Issue 2: All scores are 0.3000

**Symptoms:**
- Every post has identical score 0.3000
- Python logs show all scores defaulting

**Check:**
```bash
# 1. Check Python model loading
docker logs ctu-connect-recommend-python 2>&1 | Select-String -Pattern "loaded|PhoBERT|model"

# Should see: ✅ PhoBERT model loaded

# 2. Check user embedding generation
docker logs ctu-connect-recommend-python 2>&1 | Select-String -Pattern "user embedding|embedding for:"

# 3. Verify post embeddings exist
docker exec -it ctu-recommend-postgres psql -U postgres -d recommendation_db -c "SELECT COUNT(*) FROM post_embeddings WHERE embedding IS NOT NULL;"

# 4. Restart Python service
docker-compose restart recommend-python
```

### Issue 3: NoneType multiplication error persists

**Symptoms:**
```
ERROR: unsupported operand type(s) for *: 'NoneType' and 'float'
```

**Solution:**
```bash
# 1. Verify code changes deployed
docker exec -it ctu-connect-recommend-python cat /app/services/prediction_service.py | grep "if user_embedding is None"

# Should see the validation code

# 2. Rebuild Python container
cd d:\LVTN\CTU-Connect-demo
docker-compose build recommend-python
docker-compose up -d recommend-python

# 3. Check logs again
docker logs -f ctu-connect-recommend-python
```

---

## Success Metrics

After completing all tests:

- [ ] ✅ 5+ user_feedback records in database
- [ ] ✅ Post engagement metrics (likes, comments) updating
- [ ] ✅ Recommendation scores are diverse (different values)
- [ ] ✅ New posts get embeddings with score > 0
- [ ] ✅ No Kafka deserialization errors in logs
- [ ] ✅ No NoneType multiplication errors in logs
- [ ] ✅ API response time < 500ms for feed requests
- [ ] ✅ Cache invalidation working (can verify in Redis)

---

## Next Steps After Success

1. **Performance Monitoring**: Set up metrics tracking
2. **A/B Testing**: Compare old vs new recommendation logic
3. **User Feedback Loop**: Monitor CTR and engagement rates
4. **Model Retraining**: Schedule periodic model updates
5. **Scaling**: Consider adding more Python workers if needed

---

## Rollback if Needed

```bash
# Stop services
docker-compose stop recommend-service recommend-python

# Revert code
git checkout HEAD -- recommend-service/

# Rebuild
docker-compose up -d --build recommend-service recommend-python
```

---

## Contact

For issues or questions during testing, check:
1. This guide's troubleshooting section
2. `CRITICAL-FIXES-APPLIED.md` for detailed fixes
3. Service logs with docker logs commands above
4. Database state with SQL queries above
