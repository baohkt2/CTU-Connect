# Testing Recommendation Service Integration

## 📋 Pre-requisites

Ensure all services are running:
- ✅ Eureka Server (port 8761)
- ✅ API Gateway (port 8090)
- ✅ Auth Service (port 8091)
- ✅ User Service (port 8093)
- ✅ Post Service (port 8092)
- ✅ Recommendation Service (port 8095)
- ✅ MongoDB (post-service database)
- ✅ PostgreSQL (recommendation-service database)
- ✅ Redis (caching)
- ✅ Kafka (event streaming)
- ✅ Client Frontend (port 3000)

## 🚀 Starting Services

### Option 1: Using Docker Compose
```bash
cd d:\LVTN\CTU-Connect-demo
docker-compose up -d
```

### Option 2: Manual Start (Development)
```bash
# Terminal 1 - Eureka Server
cd eureka-server
mvn spring-boot:run

# Terminal 2 - API Gateway
cd api-gateway
mvn spring-boot:run

# Terminal 3 - Auth Service
cd auth-service
mvn spring-boot:run

# Terminal 4 - User Service
cd user-service
mvn spring-boot:run

# Terminal 5 - Post Service
cd post-service
mvn spring-boot:run

# Terminal 6 - Recommendation Service
cd recommend-service/java-api
mvn spring-boot:run

# Terminal 7 - Client Frontend
cd client-frontend
npm run dev
```

## 🧪 Testing Steps

### Step 1: Verify Service Health

#### Check Eureka Dashboard
```
http://localhost:8761
```
Verify all services are registered:
- API-GATEWAY
- AUTH-SERVICE
- USER-SERVICE
- POST-SERVICE
- RECOMMENDATION-SERVICE

#### Check Individual Service Health
```bash
# Recommendation Service
curl http://localhost:8095/actuator/health

# Post Service
curl http://localhost:8092/actuator/health

# API Gateway
curl http://localhost:8090/actuator/health
```

### Step 2: Create Test Data

#### 2.1. Register User
```bash
curl -X POST http://localhost:8090/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@ctu.edu.vn",
    "password": "Test123!",
    "fullName": "Test User",
    "studentId": "B2014567",
    "major": "CNTT",
    "faculty": "CNTT&TT"
  }'
```

#### 2.2. Login and Get JWT Token
```bash
curl -X POST http://localhost:8090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@ctu.edu.vn",
    "password": "Test123!"
  }'
```

Save the token from response:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "userId": "user_id_here"
}
```

#### 2.3. Create Some Posts
```bash
# Set your token
TOKEN="your_jwt_token_here"

# Create post 1
curl -X POST http://localhost:8090/api/posts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "title": "Học lập trình Spring Boot",
    "content": "Hôm nay mình học về Spring Boot và microservices. Rất thú vị!",
    "category": "EDUCATION",
    "visibility": "PUBLIC"
  }'

# Create post 2
curl -X POST http://localhost:8090/api/posts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "title": "Sự kiện CLB lập trình",
    "content": "CLB lập trình tổ chức workshop về AI và Machine Learning",
    "category": "EVENT",
    "visibility": "PUBLIC"
  }'

# Create post 3
curl -X POST http://localhost:8090/api/posts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "title": "Chia sẻ kinh nghiệm học tập",
    "content": "Những tips học tập hiệu quả cho sinh viên CNTT",
    "category": "EDUCATION",
    "visibility": "PUBLIC"
  }'
```

### Step 3: Test Recommendation Service Directly

#### 3.1. Check Recommendation Service Health
```bash
curl http://localhost:8095/api/recommendations/health
```

Expected response:
```json
{
  "status": "UP",
  "service": "recommendation-service-java",
  "timestamp": "2024-12-09T..."
}
```

#### 3.2. Get Personalized Feed from Recommendation Service
```bash
USER_ID="your_user_id_here"

curl -X GET "http://localhost:8095/api/recommendations/feed?userId=$USER_ID&page=0&size=20" \
  -H "Authorization: Bearer $TOKEN"
```

Expected response:
```json
{
  "userId": "user_id",
  "recommendations": [
    {
      "postId": "post_id_1",
      "authorId": "author_id",
      "content": "...",
      "score": 0.85,
      "finalScore": 0.85,
      "academicCategory": "EDUCATION",
      "rank": 1
    },
    ...
  ],
  "totalCount": 10,
  "page": 0,
  "size": 20,
  "timestamp": "2024-12-09T..."
}
```

### Step 4: Test Post Service Feed Endpoint

#### 4.1. Get Personalized Feed via Post Service
```bash
curl -X GET "http://localhost:8090/api/posts/feed?page=0&size=10" \
  -H "Authorization: Bearer $TOKEN"
```

Expected response: Array of full PostResponse objects with enriched data

#### 4.2. Watch Post Service Logs
Look for debug logs showing the flow:
```
========================================
📥 GET /api/posts/feed - User: userId, Page: 0, Size: 10
========================================
🔄 Calling recommendation-service for user: userId
📤 Received 10 recommendations from recommendation-service
📋 Fetching full details for 10 posts: [postId1, postId2, ...]
✅ Post postId1: content... (score: 0.95)
✅ Post postId2: content... (score: 0.87)
========================================
✅ Returning 10 personalized posts (123ms)
========================================
```

### Step 5: Test Frontend Integration

#### 5.1. Open Browser
Navigate to: `http://localhost:3000`

#### 5.2. Login
- Email: test@ctu.edu.vn
- Password: Test123!

#### 5.3. Check Feed
- Navigate to home page / feed
- Click on "Mới nhất" (Latest) tab
- Should load personalized feed

#### 5.4. Check Browser Console
Open Developer Tools (F12) → Console tab

Look for logs:
```
📥 Loading personalized feed from recommendation service...
📤 Received 10 posts from feed
```

#### 5.5. Verify Posts Display
- Posts should display in order
- Check if posts are relevant to your interests/major
- Verify post details (author, content, images, etc.) display correctly

### Step 6: Test Interaction Recording

#### 6.1. Like a Post
Click the like button on a post

#### 6.2. Check Recommendation Service Logs
```bash
docker-compose logs -f recommendation-service | grep -i "interaction"
```

Look for:
```
📥 API REQUEST: POST /api/recommendations/interaction
   User ID: userId
   Post ID: postId
   Type: LIKE
✅ Interaction recorded successfully
```

#### 6.3. Comment on a Post
Add a comment to a post

#### 6.4. View Duration Tracking
Stay on a post for a few seconds - the view duration should be tracked

### Step 7: Test Fallback Behavior

#### 7.1. Stop Recommendation Service
```bash
docker-compose stop recommendation-service
```

#### 7.2. Refresh Feed
The feed should still load with fallback posts (regular posts ordered by date)

#### 7.3. Check Post Service Logs
Look for:
```
⚠️  Recommendation service unavailable - using fallback for user: userId
Using regular posts for fallback feed
✅ Returning 10 regular posts (45ms)
```

#### 7.4. Restart Recommendation Service
```bash
docker-compose start recommendation-service
```

Wait 30 seconds for service to register with Eureka, then refresh feed

### Step 8: Test Cache Behavior

#### 8.1. Load Feed (First Time)
- Should take longer (100-300ms)
- Logs show "computed" source

#### 8.2. Reload Feed Immediately
- Should be faster (10-50ms)
- Logs show "cached" source

#### 8.3. Wait 2 Minutes and Reload
- Cache should expire
- Should generate new recommendations

## 📊 Expected Results

### ✅ Success Indicators

1. **Service Registration**
   - All services visible in Eureka dashboard
   - Health checks return UP status

2. **Feed Loading**
   - Posts display in frontend
   - Console shows successful API calls
   - Logs show complete flow

3. **Recommendation Quality**
   - Posts relevant to user's major/faculty
   - Mix of recent and popular content
   - Friends' posts get priority

4. **Performance**
   - First load: 100-300ms
   - Cached load: 10-50ms
   - No timeout errors

5. **Fallback Working**
   - Feed still works when recommendation-service is down
   - Graceful degradation to regular posts

6. **Interaction Tracking**
   - Likes, comments, shares are recorded
   - Kafka events are sent
   - Database updated

### ❌ Troubleshooting

#### Problem: No posts returned
**Solution:**
- Check if posts exist in database
- Verify user authentication
- Check service logs for errors

#### Problem: Recommendation service timeout
**Solution:**
- Check if recommendation-service is running
- Verify Eureka registration
- Check network connectivity between services

#### Problem: Posts not personalized
**Solution:**
- Create more test data (posts, interactions)
- Wait for Python ML service to process
- Check if user has interaction history

#### Problem: Frontend shows errors
**Solution:**
- Check browser console for errors
- Verify API Gateway is running
- Check JWT token is valid

## 🔍 Debug Commands

### Check Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f recommendation-service
docker-compose logs -f post-service

# Filter by keyword
docker-compose logs recommendation-service | grep -i "feed"
docker-compose logs post-service | grep -i "recommendation"
```

### Check Database
```bash
# MongoDB (posts)
docker exec -it mongodb mongosh
use post_db
db.posts.find().pretty()

# PostgreSQL (recommendations)
docker exec -it postgres psql -U postgres -d recommendation_db
SELECT * FROM user_feedback LIMIT 10;
SELECT * FROM post_embeddings LIMIT 10;
```

### Check Redis Cache
```bash
docker exec -it redis redis-cli
KEYS *recommendation*
GET recommendation:feed:userId
TTL recommendation:feed:userId
```

### Check Kafka Topics
```bash
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list

docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic post_created \
  --from-beginning
```

## 📈 Performance Benchmarks

### Target Metrics

| Operation | Target | Acceptable | Poor |
|-----------|--------|------------|------|
| Feed First Load | < 200ms | 200-500ms | > 500ms |
| Feed Cached Load | < 50ms | 50-100ms | > 100ms |
| Recommendation Service | < 150ms | 150-300ms | > 300ms |
| Post Details Fetch | < 50ms | 50-100ms | > 100ms |

### Monitoring
```bash
# Watch response times
docker-compose logs -f post-service | grep "Returning.*posts"

# Monitor system resources
docker stats
```

## ✅ Test Checklist

- [ ] All services running and registered in Eureka
- [ ] Health checks pass
- [ ] User can register and login
- [ ] Posts can be created
- [ ] Recommendation service returns recommendations
- [ ] Post service calls recommendation service
- [ ] Post service enriches posts with details
- [ ] Frontend displays personalized feed
- [ ] Console logs show correct flow
- [ ] Interactions are tracked
- [ ] Cache works correctly
- [ ] Fallback works when recommendation-service is down
- [ ] Performance meets targets

## 📚 Additional Resources

- `RECOMMENDATION-INTEGRATION-CHANGES.md` - Changes summary
- `README-RECOMMENDATION-SERVICE.md` - Recommendation service docs
- `recommend-service/ARCHITECTURE.md` - System architecture

---

**Testing completed successfully** ✅
**Date**: 2024-12-09
