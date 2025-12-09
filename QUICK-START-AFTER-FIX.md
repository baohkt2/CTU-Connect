# Quick Start Guide - After Critical Fixes

## 🚀 Khởi Động Nhanh (5 phút)

### 1. Restart Services
```powershell
cd d:\LVTN\CTU-Connect-demo
.\stop-all-services.ps1
Start-Sleep -Seconds 10
.\start-all-services.ps1
```

### 2. Đợi Services Khởi Động (2 phút)
```powershell
# Check status mỗi 30 giây
while ($true) {
    Write-Host "`n=== Services Status ===" -ForegroundColor Yellow
    docker-compose ps
    Start-Sleep -Seconds 30
}
```

### 3. Verify Health
```powershell
# Sau 2 phút, check health
curl http://localhost:8095/actuator/health  # Recommend-service
curl http://localhost:8000/health           # Python model
curl http://localhost:8094/actuator/health  # Post-service

# All should return: {"status":"UP"} or {"status":"healthy"}
```

---

## ✅ Quick Test (3 phút)

### Test 1: Get Feed
```powershell
$userId = "31ba8a23-8a4e-4b24-99c2-0d768e617e71"
curl "http://localhost:8095/api/recommendations/feed?userId=$userId&page=0&size=5" | ConvertFrom-Json | Select-Object -ExpandProperty content | Select-Object postId, score | Format-Table

# ✅ MUST SEE: Different scores (not all 0.3000)
# postId                       score
# ------                       -----
# 69379a305a8af849a3a4ede6    0.6543
# 6937b6b1b68143159ae33783    0.5234
# 6937c00b9bb8191d64875b31    0.4567
```

### Test 2: Like a Post
```
1. Open browser: http://localhost:3000
2. Login
3. Like any post
4. Check console below ↓
```

```powershell
# Check post-service published event
docker-compose logs --tail=20 post-service | Select-String "Published user_action"
# ✅ MUST SEE: "📤 Published user_action event: LIKE"

# Check recommend-service received event  
docker-compose logs --tail=20 recommend-service | Select-String "Received user_action"
# ✅ MUST SEE: "📥 Received user_action: LIKE"
# ✅ MUST SEE: "💾 Saved user feedback"
```

### Test 3: Verify Database
```sql
-- Run in PostgreSQL (via DataGrip or psql)
SELECT user_id, post_id, feedback_type, created_at 
FROM user_feedback 
ORDER BY created_at DESC 
LIMIT 3;

-- MUST have rows if you liked posts ↑
```

---

## ❌ If Something Goes Wrong

### Error: Bean Definition
```
BeanDefinitionOverrideException: 'userActionConsumerFactory'
```
**Fix:**
```powershell
# Verify file was saved
Get-Content "recommend-service\java-api\src\main\java\vn\ctu\edu\recommend\config\KafkaConfig.java" | Select-String "userActionConsumerFactory"
# MUST: No results

# If found, manually remove the duplicate bean definition
```

### Error: Scores All 0.3000
```
[ 1] postId1 -> score: 0.3000
[ 2] postId2 -> score: 0.3000
```
**Debug:**
```powershell
# Check Python errors
docker-compose logs recommend-service | Select-String "ERROR" | Select-Object -Last 10

# Rebuild if needed
docker-compose down
docker-compose build recommend-service --no-cache
docker-compose up -d
```

### Error: User Actions Not Received
```
Post-service: ✅ Published
Recommend-service: ❌ Not received
```
**Fix:**
```powershell
# Check Kafka
docker exec -it ctu-connect-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic user_action --from-beginning --max-messages 1

# Restart both
docker-compose restart post-service recommend-service
```

---

## 📋 Expected Log Patterns

### ✅ GOOD - Recommend-Service Startup
```
Started RecommendServiceApplication in X.XXX seconds
PhoBERT model loaded
All models loaded successfully
```

### ✅ GOOD - Feed Request
```
API REQUEST: GET /api/recommendations/feed
Post 69379a305a8af849a3a4ede6 scores: content_sim=0.6234, implicit_fb=0.7500, academic=0.2000, popularity=0.4321
Post 6937b6b1b68143159ae33783 scores: content_sim=0.4512, implicit_fb=0.5000, academic=0.1500, popularity=0.2145
```

### ✅ GOOD - User Action
```
📤 Published user_action event: LIKE by user ... on post ...
📥 Received user_action: LIKE by user ... on post ...
💾 Saved user feedback: ... -> ... (type: LIKE, value: 1.0)
📊 Updated engagement for post ...: likes=5, comments=2, ...
```

### ❌ BAD - Need Investigation
```
ERROR: unsupported operand type(s) for *: 'NoneType' and 'float'
ERROR: Can't deserialize data from topic [user_action]
BeanDefinitionOverrideException
All scores are 0.0000 or identical
```

---

## 🎯 Success Checklist

- [ ] Services start without errors
- [ ] Feed returns posts with diverse scores
- [ ] Like action appears in post-service logs
- [ ] Like action appears in recommend-service logs
- [ ] user_feedback table has new rows
- [ ] Scores change after interactions (cache invalidated)

---

**Need Help?** Check `FINAL-FIX-SUMMARY-DEC-9.md` for detailed troubleshooting.

**Estimated Time:** 10 minutes total  
**Last Updated:** December 9, 2025
