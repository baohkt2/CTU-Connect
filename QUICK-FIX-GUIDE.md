# Quick Fix Guide - Recommendation Service Integration

## 🐛 Common Issues & Solutions

### 1. Python Model Service - 422 Unprocessable Entity

**Error**: `POST /api/model/predict HTTP/1.1" 422 Unprocessable Entity`

**Cause**: Schema mismatch between Java request and Python expected schema

**Fix Applied**: 
- Changed `UserAcademicProfile.userId` from required to optional
- Location: `recommend-service/python-model/models/schemas.py`

```python
# Before
userId: str  # Required

# After
userId: Optional[str] = None  # Optional
```

**How to Apply**:
```bash
# Navigate to python-model directory
cd d:\LVTN\CTU-Connect-demo\recommend-service\python-model

# Restart the service
# If running via PowerShell script:
.\run-dev.ps1

# Or if running directly:
python server.py
```

---

### 2. Python Model Service - ModuleNotFoundError: No module named 'app'

**Error**: `ModuleNotFoundError: No module named 'app'`

**Fix Applied**: Singleton pattern for PredictionService
- Location: `recommend-service/python-model/api/routes.py`
- See: `BUGFIX-PREDICTION-SERVICE.md` for details

**Status**: ✅ Already Fixed

---

### 3. Post Service - RecommendationServiceClient Not Found

**Error**: Service fails to call recommendation-service

**Check**:
```bash
# 1. Verify Eureka registration
curl http://localhost:8761

# 2. Check if recommendation-service is up
curl http://localhost:8095/actuator/health

# 3. Check post-service logs
docker-compose logs -f post-service | grep -i "recommendation"
```

**Solution**: Ensure all services are registered with Eureka

---

### 4. Frontend - Posts Not Loading

**Error**: Blank feed or error in console

**Check Browser Console**:
```javascript
// Look for:
📥 Loading personalized feed from recommendation service...
❌ Error loading posts: ...
```

**Solutions**:
1. Check if user is authenticated (JWT token valid)
2. Verify API Gateway is running (port 8090)
3. Check post-service is running (port 8092)
4. Look at Network tab for failed requests

---

## 🚀 Quick Restart Services

### Restart All Services
```bash
cd d:\LVTN\CTU-Connect-demo

# Stop all
docker-compose down

# Start all
docker-compose up -d

# Watch logs
docker-compose logs -f post-service recommendation-service
```

### Restart Specific Service

**Recommendation Service (Java)**:
```bash
docker-compose restart recommendation-service
```

**Python Model Service**:
```bash
# Stop current process (Ctrl+C)
cd d:\LVTN\CTU-Connect-demo\recommend-service\python-model
python server.py
```

**Post Service**:
```bash
docker-compose restart post-service
```

**Client Frontend**:
```bash
cd d:\LVTN\CTU-Connect-demo\client-frontend
npm run dev
```

---

## 🔍 Debug Checklist

### Before Testing
- [ ] All Docker containers running: `docker-compose ps`
- [ ] Eureka shows all services: `http://localhost:8761`
- [ ] Python service responding: `curl http://localhost:5000/health`
- [ ] Recommendation service healthy: `curl http://localhost:8095/actuator/health`
- [ ] Post service healthy: `curl http://localhost:8092/actuator/health`

### Test Feed Endpoint
```bash
# 1. Get JWT token
TOKEN=$(curl -X POST http://localhost:8090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@ctu.edu.vn","password":"password"}' \
  | jq -r '.token')

# 2. Test feed endpoint
curl -X GET "http://localhost:8090/api/posts/feed?page=0&size=10" \
  -H "Authorization: Bearer $TOKEN"

# 3. Check logs
docker-compose logs --tail=50 post-service | grep "feed"
```

### Expected Logs

**Post Service**:
```
========================================
📥 GET /api/posts/feed - User: userId, Page: 0, Size: 10
========================================
🔄 Calling recommendation-service for user: userId
📤 Received X recommendations from recommendation-service
📋 Fetching full details for X posts
✅ Returning X personalized posts (XXXms)
========================================
```

**Recommendation Service**:
```
📥 API REQUEST: GET /api/recommendations/feed
   User ID: userId
   Page: 0, Size: 10
🔄 Calling hybrid recommendation service
✅ Generated recommendations successfully
📤 API RESPONSE: X recommendations
```

---

## 🆘 If Nothing Works

### Nuclear Option - Full Reset
```bash
# 1. Stop everything
docker-compose down -v

# 2. Clean volumes
docker volume prune -f

# 3. Rebuild
docker-compose build --no-cache

# 4. Start fresh
docker-compose up -d

# 5. Wait for services to be ready (30-60 seconds)
sleep 60

# 6. Check Eureka
curl http://localhost:8761
```

### Check Service Dependencies
```bash
# MongoDB for post-service
docker-compose ps mongodb

# PostgreSQL for recommendation-service  
docker-compose ps postgres

# Redis for caching
docker-compose ps redis

# Kafka for events
docker-compose ps kafka
```

---

## 📞 Get Help

### Check Documentation
1. `IMPLEMENTATION-SUMMARY.md` - Overview
2. `TEST-RECOMMENDATION-FLOW.md` - Detailed testing
3. `RECOMMENDATION-INTEGRATION-CHANGES.md` - Technical changes
4. `BUGFIX-PREDICTION-SERVICE.md` - Python service fix

### Debug Commands
```bash
# View all logs
docker-compose logs -f

# Search logs
docker-compose logs post-service | grep -i "error"
docker-compose logs recommendation-service | grep -i "exception"

# Check Java heap
docker stats recommendation-service

# Test direct endpoints
curl http://localhost:8092/actuator/health
curl http://localhost:8095/actuator/health
curl http://localhost:5000/health
```

---

**Last Updated**: December 9, 2024  
**Status**: Active Issues Tracked & Fixed
