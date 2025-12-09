# Quick Test Checklist ✅

**After deploying fixes, run through this checklist:**

---

## 🚀 Deploy (5 minutes)

```bash
# Build
cd d:\LVTN\CTU-Connect-demo\recommend-service\java-api
mvn clean package -DskipTests

# Restart
cd ..\..
docker-compose up -d --build recommend-service recommend-python

# Verify running
docker ps | findstr recommend
```

**Expected**: 4 containers (recommend-service, recommend-python, postgres, redis) all UP

---

## 📝 Test 1: Like Post (2 minutes)

### Action
```http
POST http://localhost:8080/api/posts/{postId}/interactions
{ "reaction": "LIKE" }
```

### Check Logs
```bash
docker logs --tail 20 ctu-connect-recommend-service | Select-String "user_action|Received"
```

### Verify Database
```sql
docker exec -it ctu-recommend-postgres psql -U postgres -d recommendation_db -c "SELECT COUNT(*) FROM user_feedback WHERE feedback_type='LIKE';"
```

**Expected Result**: Count increases by 1

✅ Pass  ❌ Fail

---

## 📝 Test 2: Get Recommendations (2 minutes)

### Action
```http
GET http://localhost:8095/api/recommendations/feed?userId={userId}&page=0&size=10
```

### Check Python Logs
```bash
docker logs --tail 30 ctu-connect-recommend-python | Select-String "Prediction|scores"
```

### Check Response
- Are scores **different** for different posts? (Not all 0.3000)
- Do you see contentSimilarity, implicitFeedback, academicScore, popularityScore?

**Expected**: Diverse scores like 0.4523, 0.2891, 0.6789

✅ Pass  ❌ Fail

---

## 📝 Test 3: Create Post (2 minutes)

### Action
```http
POST http://localhost:8080/api/posts
{
  "content": "Test post for embedding",
  "category": "GENERAL"
}
```

### Check Logs
```bash
docker logs --tail 20 ctu-connect-recommend-service | Select-String "POST_CREATED|Embedding"
```

### Verify Database
```sql
docker exec -it ctu-recommend-postgres psql -U postgres -d recommendation_db -c "SELECT post_id, score FROM post_embeddings ORDER BY created_at DESC LIMIT 1;"
```

**Expected Result**: New row with score > 0.0

✅ Pass  ❌ Fail

---

## 📝 Test 4: No Errors (1 minute)

### Check for Errors
```bash
# Java errors
docker logs --tail 50 ctu-connect-recommend-service 2>&1 | Select-String "ERROR|Exception"

# Python errors  
docker logs --tail 50 ctu-connect-recommend-python 2>&1 | Select-String "ERROR|NoneType"
```

**Expected**: No deserialization errors, no NoneType errors

✅ Pass  ❌ Fail

---

## 📝 Test 5: Database Health (1 minute)

```sql
docker exec -it ctu-recommend-postgres psql -U postgres -d recommendation_db

-- Should have data
SELECT COUNT(*) FROM user_feedback;           -- > 0
SELECT COUNT(*) FROM post_embeddings;         -- > 0
SELECT AVG(score) FROM post_embeddings;       -- > 0.0
SELECT AVG(popularity_score) FROM post_embeddings WHERE like_count > 0;  -- > 0.0

\q
```

✅ Pass  ❌ Fail

---

## 📊 Final Score

**Tests Passed**: ___ / 5

- **5/5**: ✅ Perfect! System working as expected
- **4/5**: ⚠️  Good, minor issue to investigate
- **3/5**: ⚠️  Partial success, needs attention
- **< 3**: ❌ Investigate failed tests, check logs

---

## 🔍 If Any Test Fails

### Quick Debug

```bash
# 1. Check all services running
docker ps

# 2. Check Kafka connectivity
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list

# 3. Check messages in topic
docker exec -it kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic user_action --from-beginning --max-messages 3

# 4. Restart services
docker-compose restart recommend-service recommend-python

# 5. Re-run tests
```

---

## 📚 Full Documentation

- **FIX-SUMMARY-2025-12-09.md** - Executive summary
- **CRITICAL-FIXES-APPLIED.md** - Technical details of each fix
- **RESTART-AND-TEST-GUIDE.md** - Complete testing procedures

---

## ✅ Success Indicators

Look for these in logs:

```
✅ Successfully processed user_action event: LIKE
✅ Embedding saved successfully
✅ PhoBERT model loaded
```

Avoid seeing these:

```
❌ No serializer found for class
❌ unsupported operand type(s) for *: 'NoneType'
❌ Error parsing user action event
```

---

**Time to Complete**: ~13 minutes  
**Difficulty**: Easy  
**Prerequisites**: Services deployed and running
