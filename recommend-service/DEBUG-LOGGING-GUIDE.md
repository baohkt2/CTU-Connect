# Debug Logging Guide - Recommendation Service

## 📊 Overview

Enhanced debug logging đã được thêm vào recommend-service để tracking chi tiết list posts được recommend kèm scores.

## 🔍 Log Format

### 1. API Request/Response Log (RecommendationController)

```
========================================
📥 API REQUEST: GET /api/recommendations/feed
   User ID: 675656b4c82ce14ba0000001
   Page: 0, Size: 20
========================================
```

### 2. Recommended Posts List (Chi tiết từng post)

```
📋 RECOMMENDED POSTS LIST:
   Format: [Rank] PostID -> Score
   ----------------------------------------
   [ 1] 67885a2e9f1234567890abcd -> score: 0.9543
   [ 2] 67885a2e9f1234567890abce -> score: 0.9123
   [ 3] 67885a2e9f1234567890abcf -> score: 0.8876
   [ 4] 67885a2e9f1234567890abd0 -> score: 0.8654
   [ 5] 67885a2e9f1234567890abd1 -> score: 0.8432
   ...
   ----------------------------------------
```

### 3. Score Statistics

```
📊 SCORE STATISTICS:
   Max Score: 0.9543
   Min Score: 0.5234
   Avg Score: 0.7845
```

### 4. Processing Stages

#### Stage 1: Python Model Rankings
```
🤖 Python model returned 40 ranked posts
📊 PYTHON MODEL RANKINGS (Top 5):
   • PostID: 67885a2e9f1234567890abcd | ML Score: 0.9123 | Category: EDUCATION
   • PostID: 67885a2e9f1234567890abce | ML Score: 0.8876 | Category: EVENT
   • PostID: 67885a2e9f1234567890abcf | ML Score: 0.8654 | Category: EDUCATION
   ...
```

#### Stage 2: After Python/Fallback
```
📦 AFTER PYTHON/FALLBACK (before business rules):
   Total: 40 posts
   • PostID: 67885a2e9f1234567890abcd | Score: 0.9123
   • PostID: 67885a2e9f1234567890abce | Score: 0.8876
   • PostID: 67885a2e9f1234567890abcf | Score: 0.8654
```

#### Stage 3: After Business Rules
```
⚖️  AFTER BUSINESS RULES:
   Total: 40 posts
   • PostID: 67885a2e9f1234567890abcd | Adjusted Score: 0.9543  (+0.3 friend boost)
   • PostID: 67885a2e9f1234567890abce | Adjusted Score: 0.9123  (+0.2 major boost)
   • PostID: 67885a2e9f1234567890abcf | Adjusted Score: 0.8876
```

#### Stage 4: Final Recommendations
```
🎯 FINAL RECOMMENDATIONS (before caching):
   Total: 20 posts
   Top 5 Posts:
   • PostID: 67885a2e9f1234567890abcd | Score: 0.9543 | Category: EDUCATION
   • PostID: 67885a2e9f1234567890abce | Score: 0.9123 | Category: EVENT
   • PostID: 67885a2e9f1234567890abcf | Score: 0.8876 | Category: EDUCATION
   • PostID: 67885a2e9f1234567890abd0 | Score: 0.8654 | Category: ANNOUNCEMENT
   • PostID: 67885a2e9f1234567890abd1 | Score: 0.8432 | Category: EDUCATION
```

## 📝 How to View Logs

### Method 1: Docker Compose Logs
```bash
# View recommend-service logs in real-time
docker-compose logs -f recommendation-service

# Filter for specific patterns
docker-compose logs recommendation-service | grep "RECOMMENDED POSTS"
docker-compose logs recommendation-service | grep "SCORE STATISTICS"
```

### Method 2: Direct Service Logs
```bash
# If running locally
cd d:\LVTN\CTU-Connect-demo\recommend-service\java-api
mvn spring-boot:run

# Logs will appear in console
```

### Method 3: Log File
```bash
# Check application logs directory
cd d:\LVTN\CTU-Connect-demo\recommend-service\java-api\logs
tail -f application.log
```

## 🧪 Testing Debug Logs

### Step 1: Trigger Recommendation Request
```bash
# Get JWT token first
TOKEN="your_jwt_token_here"

# Request personalized feed
curl -X GET "http://localhost:8090/api/posts/feed?page=0&size=10" \
  -H "Authorization: Bearer $TOKEN"
```

### Step 2: Watch Logs
```bash
# In another terminal
docker-compose logs -f recommendation-service
```

### Step 3: Expected Output
```
========================================
📥 API REQUEST: GET /api/recommendations/feed
   User ID: 675656b4c82ce14ba0000001
   Page: 0, Size: 10
========================================
🔄 Calling hybrid recommendation service for feed generation
Getting feed for user: 675656b4c82ce14ba0000001, size: 10
User profile: major=CNTT, faculty=CNTT&TT
User has 5 interactions in last 30 days
Found 50 candidate posts
🤖 Python model returned 40 ranked posts
📊 PYTHON MODEL RANKINGS (Top 5):
   • PostID: 67885a2e9f1234567890abcd | ML Score: 0.9123 | Category: EDUCATION
   • PostID: 67885a2e9f1234567890abce | ML Score: 0.8876 | Category: EVENT
   ...
📦 AFTER PYTHON/FALLBACK (before business rules):
   Total: 40 posts
   ...
⚖️  AFTER BUSINESS RULES:
   Total: 40 posts
   ...
🎯 FINAL RECOMMENDATIONS (before caching):
   Total: 10 posts
   Top 5 Posts:
   • PostID: 67885a2e9f1234567890abcd | Score: 0.9543 | Category: EDUCATION
   ...
========================================
📤 API RESPONSE: GET /api/recommendations/feed
   Total Items: 10
   User ID: 675656b4c82ce14ba0000001
📋 RECOMMENDED POSTS LIST:
   Format: [Rank] PostID -> Score
   ----------------------------------------
   [ 1] 67885a2e9f1234567890abcd -> score: 0.9543
   [ 2] 67885a2e9f1234567890abce -> score: 0.9123
   [ 3] 67885a2e9f1234567890abcf -> score: 0.8876
   [ 4] 67885a2e9f1234567890abd0 -> score: 0.8654
   [ 5] 67885a2e9f1234567890abd1 -> score: 0.8432
   [ 6] 67885a2e9f1234567890abd2 -> score: 0.8210
   [ 7] 67885a2e9f1234567890abd3 -> score: 0.7998
   [ 8] 67885a2e9f1234567890abd4 -> score: 0.7765
   [ 9] 67885a2e9f1234567890abd5 -> score: 0.7543
   [10] 67885a2e9f1234567890abd6 -> score: 0.7321
   ----------------------------------------
📊 SCORE STATISTICS:
   Max Score: 0.9543
   Min Score: 0.7321
   Avg Score: 0.8456
========================================
```

## 📋 Log Analysis Tips

### 1. Extract Post IDs and Scores
```bash
# Extract from logs
docker-compose logs recommendation-service | \
  grep "RECOMMENDED POSTS LIST" -A 30 | \
  grep -E "\[[0-9]+\]" | \
  awk '{print $2, $4}'
```

Output:
```
67885a2e9f1234567890abcd 0.9543
67885a2e9f1234567890abce 0.9123
67885a2e9f1234567890abcf 0.8876
...
```

### 2. Check Score Distribution
```bash
# Get score statistics
docker-compose logs recommendation-service | \
  grep "SCORE STATISTICS" -A 3
```

### 3. Monitor Python Model Performance
```bash
# Check if Python model is being used
docker-compose logs recommendation-service | \
  grep -E "(🤖 Python model|⚠️.*unavailable)"
```

### 4. Track Business Rule Impact
```bash
# Compare scores before and after business rules
docker-compose logs recommendation-service | \
  grep -E "(AFTER PYTHON|AFTER BUSINESS)" -A 5
```

## 🔧 Troubleshooting

### Issue: No logs appearing
**Check**:
```bash
# Verify service is running
docker-compose ps recommendation-service

# Check log level in application.properties
grep "logging.level" recommend-service/java-api/src/main/resources/application.properties
```

**Should have**:
```properties
logging.level.vn.ctu.edu.recommend=INFO
```

### Issue: Scores all 0.0000
**Possible causes**:
1. Python model service not running
2. No candidate posts found
3. Fallback ranking being used

**Check**:
```bash
# Look for warnings
docker-compose logs recommendation-service | grep -E "(⚠️|Python.*unavailable)"
```

### Issue: Empty recommendation list
**Check**:
```bash
# Look for candidate posts count
docker-compose logs recommendation-service | grep "Found.*candidate posts"

# Check database
docker exec -it postgres psql -U postgres -d recommendation_db
SELECT COUNT(*) FROM post_embeddings;
```

## 📊 Sample Analysis

### Scenario 1: Good Recommendations
```
Total: 10 posts
Max Score: 0.9543   ← High confidence
Min Score: 0.7321   ← Still good quality
Avg Score: 0.8456   ← Strong overall
```
✅ System is working well

### Scenario 2: Poor Recommendations
```
Total: 10 posts
Max Score: 0.4523   ← Low confidence
Min Score: 0.1221   ← Very low quality
Avg Score: 0.2845   ← Weak overall
```
⚠️ Need to investigate:
- User interaction history too sparse
- Not enough candidate posts
- Python model not trained properly

### Scenario 3: Fallback Mode
```
⚠️  Python model service unavailable, using fallback ranking
📦 AFTER PYTHON/FALLBACK (before business rules):
   Total: 40 posts
```
⚠️ Python service issue - check python-model logs

## 🎯 Best Practices

1. **Always check full log flow** from API request to response
2. **Compare scores** at different stages to understand boosts
3. **Monitor Python model usage** to ensure ML is active
4. **Track score statistics** to measure recommendation quality
5. **Save logs for analysis** when testing new features

## 📚 Related Files

- `RecommendationController.java` - API endpoint logging
- `HybridRecommendationService.java` - Processing stage logging
- `application.properties` - Log level configuration

---

**Created**: December 9, 2024  
**Purpose**: Track and debug recommendation scoring  
**Format**: {postId, final_score} with detailed stages
