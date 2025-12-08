# CTU Connect - Recommendation Service Integration

## 🎯 Tổng quan

Đã hoàn thành cài đặt và tích hợp **Recommendation Service** - hệ thống gợi ý cá nhân hóa sử dụng AI cho CTU Connect.

## ✅ Đã hoàn thành

### 1. Service Communication (Feign Clients)
- **PostServiceClient**: Gọi Post Service để lấy thông tin posts
- **UserServiceFeignClient**: Gọi User Service để lấy thông tin user và academic profile
- **PythonModelServiceClient**: Gọi Python ML service để ranking
- Fallback handlers cho tất cả clients (circuit breaker pattern)

### 2. API Gateway Routes
```java
/api/recommendations/** → recommendation-service
/api/feed/**           → recommendation-service
```

### 3. Kafka Event Integration
Post Service gửi events đến Recommendation Service:
- `post_created` - Tạo embedding cho post mới
- `post_updated` - Update embedding khi content thay đổi
- `post_deleted` - Xóa embedding và cache
- `user_action` - Thu thập user interactions (like, comment, share, view)

### 4. Configuration
- Feign circuit breaker enabled
- JWT token auto-forwarding
- Application profiles (dev, docker)
- Redis caching strategy
- Kafka consumer groups

## 📁 Files đã tạo/sửa

### Recommendation Service
```
recommend-service/
├── java-api/src/main/java/vn/ctu/edu/recommend/
│   ├── client/
│   │   ├── PostServiceClient.java (NEW)
│   │   ├── PostServiceClientFallback.java (NEW)
│   │   ├── UserServiceFeignClient.java (NEW)
│   │   └── UserServiceFeignClientFallback.java (NEW)
│   ├── config/
│   │   └── FeignConfig.java (NEW)
│   └── model/dto/
│       ├── PostDTO.java (NEW)
│       └── UserDTO.java (NEW)
├── INTEGRATION_GUIDE.md (NEW)
├── SETUP_COMPLETE.md (NEW)
├── QUICK_START.md (NEW)
└── test-integration.ps1 (NEW)
```

### Post Service
```
post-service/
└── src/main/java/com/ctuconnect/service/
    └── EventService.java (UPDATED)
        - Thêm publishing events cho recommendation service
```

### API Gateway
```
api-gateway/
└── src/main/java/com/ctuconnect/config/
    └── RouteConfig.java (UPDATED)
        - Thêm routes cho recommendation service
```

## 🚀 Khởi động hệ thống

### 1. Start tất cả services
```bash
docker-compose up -d
```

### 2. Kiểm tra services
```bash
# Test integration
powershell -ExecutionPolicy Bypass -File .\recommend-service\test-integration.ps1

# Hoặc test thủ công
curl http://localhost:8761/eureka/apps/RECOMMENDATION-SERVICE
curl http://localhost:8095/actuator/health
```

### 3. Xem logs
```bash
docker-compose logs -f recommendation-service
```

## 🔗 Kiến trúc tích hợp

```
                    ┌─────────────┐
                    │ API Gateway │
                    │  Port 8090  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │   Post   │      │   User   │      │ Recommend│
  │ Service  │      │ Service  │      │ Service  │
  │ :8092    │      │ :8093    │      │ :8095    │
  └────┬─────┘      └────┬─────┘      └────┬─────┘
       │                 │                  │
       │ Feign Clients ─┼──────────────────┘
       │                 │
       └─────► Kafka Events
           (post_created, user_action)
```

## 📋 API Endpoints

### Via API Gateway (http://localhost:8090)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | /api/recommendations/feed/{userId} | Personalized feed | Bearer |
| GET | /api/recommendations/similar/{postId} | Similar posts | Bearer |
| POST | /api/recommendations/feedback | Submit feedback | Bearer |

### Direct Service (http://localhost:8095)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /actuator/health | Health check |
| GET | /actuator/prometheus | Metrics |
| GET | /actuator/info | Service info |

## 🧪 Testing

### 1. Health Check
```bash
curl http://localhost:8095/actuator/health
```

Expected response:
```json
{
  "status": "UP",
  "components": {
    "db": {"status": "UP"},
    "neo4j": {"status": "UP"},
    "redis": {"status": "UP"},
    "kafka": {"status": "UP"}
  }
}
```

### 2. Test với API Gateway (cần JWT token)
```bash
# Get token from auth service first
TOKEN=$(curl -X POST http://localhost:8090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@ctu.edu.vn","password":"password"}' \
  | jq -r '.token')

# Test recommendation endpoint
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8090/api/recommendations/feed/userId
```

### 3. Test Kafka Integration
```bash
# Check consumer groups
docker exec -it kafka /opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group recommendation-service-group \
  --describe

# List topics
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list
```

## 📊 Monitoring

### Service Status
```bash
# Eureka dashboard
open http://localhost:8761

# Neo4j browser
open http://localhost:7474

# Recommendation metrics
curl http://localhost:8095/actuator/prometheus
```

### Logs
```bash
# All logs
docker-compose logs -f

# Specific service
docker-compose logs -f recommendation-service

# Filter by keyword
docker-compose logs recommendation-service | grep -i "feign"
docker-compose logs recommendation-service | grep -i "kafka"
```

## ⚠️ Cần hoàn thiện

### User Service cần implement:

#### 1. Academic Profile Endpoint
```java
@GetMapping("/api/users/{userId}/academic-profile")
public ResponseEntity<UserAcademicProfile> getAcademicProfile(@PathVariable String userId) {
    return ResponseEntity.ok(UserAcademicProfile.builder()
        .userId(userId)
        .major("CNTT")
        .faculty("CNTT&TT")
        .degree("Bachelor")
        .batch("K44")
        .studentId("B1234567")
        .build());
}
```

#### 2. Friends List Endpoint
```java
@GetMapping("/api/users/{userId}/friends")
public ResponseEntity<List<String>> getUserFriends(@PathVariable String userId) {
    List<String> friendIds = userService.getFriends(userId);
    return ResponseEntity.ok(friendIds);
}
```

## 🔧 Troubleshooting

### Service không start
```bash
# Check all services
docker-compose ps

# Restart specific service
docker-compose restart recommendation-service

# Full restart
docker-compose down
docker-compose up -d
```

### Feign calls failing
```bash
# Check service registration
curl http://localhost:8761/eureka/apps

# Check network
docker network inspect ctu-connect-demo_ctuconnect-network

# Test connectivity
docker exec -it ctu-recommendation-service ping post-service
```

### Kafka issues
```bash
# Restart Kafka
docker-compose restart kafka

# Check topics
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list

# Consumer group lag
docker exec -it kafka /opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --group recommendation-service-group \
  --describe
```

## 📚 Documentation

Xem thêm chi tiết:
- **INTEGRATION_GUIDE.md** - Hướng dẫn tích hợp chi tiết
- **SETUP_COMPLETE.md** - Tổng quan các thay đổi
- **QUICK_START.md** - Quick start guide
- **ARCHITECTURE.md** - Kiến trúc hệ thống

## 🎓 Features

- ✅ **Content-based filtering**: PhoBERT embeddings cho tiếng Việt
- ✅ **Collaborative filtering**: Neo4j graph relationships
- ✅ **Academic classification**: Phân loại nội dung học thuật
- ✅ **Popularity ranking**: Ranking theo engagement
- ✅ **Real-time events**: Kafka event streaming
- ✅ **Caching**: Redis multi-level caching
- ✅ **Circuit breaker**: Resilient service communication
- ✅ **Fallback**: Graceful degradation

## 📈 Next Steps

1. ✅ **Complete**: Service integration setup
2. ✅ **Complete**: Kafka event flow
3. ✅ **Complete**: API Gateway routing
4. ⚠️ **Pending**: User Service academic profile endpoints
5. ⚠️ **Pending**: Frontend integration
6. ⚠️ **Pending**: End-to-end testing
7. ⚠️ **Pending**: Performance tuning
8. ⚠️ **Pending**: Production deployment

## 🆘 Support

Nếu gặp vấn đề:
1. Check logs: `docker-compose logs -f recommendation-service`
2. Run integration test: `.\recommend-service\test-integration.ps1`
3. Review documentation in `recommend-service/` folder
4. Check Eureka dashboard: http://localhost:8761

---

**Status**: ✅ Ready for integration testing  
**Build**: ✅ Successful  
**Date**: 2025-12-08  
**Version**: 1.0.0
