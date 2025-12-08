# Quick Start Guide - Recommendation Service

## 🚀 Start Services

### 1. Start Required Infrastructure (Docker)

```bash
cd d:\LVTN\CTU-Connect-demo

# Start all databases
docker-compose up -d eureka-server neo4j recommend-postgres recommend-redis kafka mongodb
```

### 2. Start Recommendation Service (IDE/Maven)

```bash
cd recommend-service/java-api

# Option A: Maven
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# Option B: IDE (IntelliJ/Eclipse)
# Right-click on main class -> Run
# Make sure profile is set to "dev"
```

### 3. Verify Registration

Open browser: http://localhost:8761

Look for `RECOMMENDATION-SERVICE` in registered applications.

---

## 🧪 Test API

### Run Test Script

```powershell
cd recommend-service
.\test-recommendation-api.ps1
```

### Manual Test

```bash
# Health check
curl http://localhost:8095/api/recommend/health

# Get recommendations
curl "http://localhost:8095/api/recommend/posts?userId=test-user&page=0&size=10"
```

---

## 💻 Use in Frontend

### Import Service

```typescript
import { postService } from '@/services/postService';
```

### Get Recommendations

```typescript
// Option 1: AI Recommendations
const posts = await postService.getRecommendedPosts(userId, 0, 20);

// Option 2: Personalized Feed
const feed = await postService.getPersonalizedFeed(userId, 0, 20);
```

### Track Interaction

```typescript
// When user views a post
await postService.recordRecommendationInteraction(
  userId,
  postId,
  'VIEW',
  30 // seconds
);

// When user likes a post
await postService.recordRecommendationInteraction(userId, postId, 'LIKE');
```

---

## 📊 Monitor Logs

Logs will show detailed API information:

```
========================================
📥 API REQUEST: GET /api/recommend/posts
   User ID: user123
   Page: 0, Size: 20
========================================
🔄 Processing recommendation request...
========================================
📤 API RESPONSE: Success
   Total Recommendations: 20
   Algorithm: HYBRID_ML
========================================
```

---

## 🔧 Troubleshooting

### Service Not Registered?

Check `application-dev.yml`:
```yaml
eureka:
  client:
    enabled: true  # Must be true!
```

### 404 Not Found?

Verify endpoint:
- ✅ `/api/recommend/posts` → Recommendation Service
- ✅ `/api/recommendation/feed` → Recommendation Service
- ✅ `/api/recommendations/...` → Post Service (legacy)

### Empty Results?

- Check Neo4j has data
- Check PostgreSQL has posts
- Check Redis is accessible

---

## 📖 Full Documentation

- **API Flow:** [API-FLOW-DOCUMENTATION.md](./API-FLOW-DOCUMENTATION.md)
- **Changes:** [CHANGES-SUMMARY.md](./CHANGES-SUMMARY.md)
- **Setup:** [README-SETUP.md](./python-model/README-SETUP.md)

---

## 🎯 Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/recommend/posts` | AI recommendations |
| `GET /api/recommendation/feed` | Personalized feed |
| `POST /api/recommendation/interaction` | Track interaction |
| `POST /api/recommend/feedback` | User feedback |
| `GET /api/recommend/health` | Health check |

---

## ✅ Checklist

Before testing:

- [ ] Eureka Server running (8761)
- [ ] API Gateway running (8090)
- [ ] Recommendation Service running (8095)
- [ ] Neo4j running (7687)
- [ ] PostgreSQL running (5435)
- [ ] Redis running (6380)
- [ ] Service registered in Eureka

---

**Ready to code!** 🎉
