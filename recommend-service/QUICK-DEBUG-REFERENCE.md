# Quick Debug Reference - Recommendation Service

## 🚀 Quick Commands

### View Real-Time Logs
```bash
# Watch recommendation service logs
docker-compose logs -f recommendation-service

# Filter for post list
docker-compose logs recommendation-service | grep "RECOMMENDED POSTS LIST" -A 25

# Filter for scores
docker-compose logs recommendation-service | grep "SCORE STATISTICS" -A 5
```

### Extract Post IDs and Scores
```bash
# Get PostID -> Score mapping
docker-compose logs recommendation-service | \
  grep -E "\[[0-9]+\].*->" | \
  awk '{print $2, $4}'
```

### Test Recommendation Endpoint
```bash
# 1. Get token
TOKEN=$(curl -s -X POST http://localhost:8090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@ctu.edu.vn","password":"password"}' | \
  jq -r '.token')

# 2. Get recommendations
curl -X GET "http://localhost:8090/api/posts/feed?page=0&size=10" \
  -H "Authorization: Bearer $TOKEN" | jq

# 3. Watch logs in another terminal
docker-compose logs -f recommendation-service
```

## 📊 Log Patterns to Watch

### ✅ Success Pattern
```
📥 API REQUEST → 
🔄 Getting feed → 
🤖 Python model returned X posts → 
📊 PYTHON MODEL RANKINGS → 
⚖️  AFTER BUSINESS RULES → 
🎯 FINAL RECOMMENDATIONS → 
📋 RECOMMENDED POSTS LIST → 
📊 SCORE STATISTICS → 
📤 API RESPONSE ✓
```

### ⚠️ Fallback Pattern
```
📥 API REQUEST → 
🔄 Getting feed → 
⚠️  Python model unavailable → 
📦 Using fallback ranking → 
📋 RECOMMENDED POSTS LIST
```

### ❌ Error Pattern
```
📥 API REQUEST → 
❌ ERROR: GET /api/recommendations/feed failed
   Error: [error message]
```

## 🎯 What Each Log Means

| Log Marker | Meaning | Action |
|------------|---------|--------|
| 📥 | Request received | Normal |
| 🔄 | Processing started | Normal |
| 🤖 | Python ML active | ✅ Good |
| ⚠️  | Using fallback | Check Python service |
| ⚖️  | Applying boosts | Normal (friend/major) |
| 🎯 | Final results | Check count |
| 📋 | Post list output | **This is what you want** |
| 📊 | Statistics | Check quality |
| 📤 | Response sent | Success |
| ❌ | Error occurred | Investigate |

## 🔍 Quick Checks

### 1. Are recommendations working?
```bash
# Should see posts with scores
docker-compose logs recommendation-service | grep "Total Items:" | tail -1
```

### 2. Is Python ML being used?
```bash
# Should see "Python model returned"
docker-compose logs recommendation-service | grep "🤖" | tail -1
```

### 3. What are typical scores?
```bash
# Should see stats
docker-compose logs recommendation-service | grep "Max Score:" | tail -1
```

### 4. How many posts returned?
```bash
# Check post count
docker-compose logs recommendation-service | grep "Total: .* posts" | tail -3
```

## 🎨 Example Output Format

```
📋 RECOMMENDED POSTS LIST:
   Format: [Rank] PostID -> Score
   ----------------------------------------
   [ 1] 67885a2e9f123 -> score: 0.9543
   [ 2] 67885a2e9f124 -> score: 0.9123
   [ 3] 67885a2e9f125 -> score: 0.8876
   [ 4] 67885a2e9f126 -> score: 0.8654
   [ 5] 67885a2e9f127 -> score: 0.8432
   ----------------------------------------
📊 SCORE STATISTICS:
   Max Score: 0.9543
   Min Score: 0.7321
   Avg Score: 0.8456
```

## 💡 Pro Tips

1. **Grep for markers**: Use emoji markers (📋, 📊, 🎯) for quick filtering
2. **Save logs**: Redirect to file for analysis: `> debug.log`
3. **Compare stages**: Check scores at Python → Business Rules → Final
4. **Monitor changes**: Watch how scores change with boosts
5. **Track patterns**: Look for consistent high/low scorers

## 🔧 Common Issues & Quick Fixes

### Issue: Empty list
```bash
# Check candidate posts
docker-compose logs recommendation-service | grep "Found.*candidate"

# Should see: "Found X candidate posts" where X > 0
```

### Issue: All scores 0.0000
```bash
# Check Python service
docker-compose ps python-model
docker-compose logs python-model | tail -20

# Restart if needed
docker-compose restart python-model
```

### Issue: No logs appearing
```bash
# Check service status
docker-compose ps recommendation-service

# Restart service
docker-compose restart recommendation-service
```

## 📞 Need More Detail?

See full documentation: `DEBUG-LOGGING-GUIDE.md`

---

**Quick Reference** | Updated: Dec 9, 2024
