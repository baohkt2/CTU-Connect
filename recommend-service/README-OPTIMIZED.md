# CTU Connect - Recommendation Service (Optimized)

## 🎯 Overview

Recommendation Service cho hệ thống CTU Connect, cung cấp gợi ý bài viết được cá nhân hóa dựa trên AI/ML.

**Version**: 2.0.0 (Optimized - December 2024)

## 🏗️ Architecture

Hệ thống sử dụng **Hybrid Architecture**:
- **Java Service** (Port 8081): Orchestrator, Business Logic, Caching
- **Python Service** (Port 8000): AI Inference Engine (PhoBERT)

```
Frontend → Java Service → Python Service → PhoBERT Model
              ↓              ↓
           Cache         Embedding
           (Redis)       Generation
              ↓
           Database
         (PostgreSQL)
```

Chi tiết xem: [ARCHITECTURE-OPTIMIZED.md](./ARCHITECTURE-OPTIMIZED.md)

## ✨ Recent Optimizations (v2.0.0)

### What Changed?
1. ✅ **Consolidated Controllers**: Merged 2 controllers → 1 unified controller
2. ✅ **Unified Python Entry**: Merged 2 apps → 1 single entry point
3. ✅ **Clear API Pattern**: Single endpoint pattern `/api/recommendations/*`
4. ✅ **Better Separation**: Java = Orchestrator, Python = AI only
5. ✅ **Removed Duplicates**: ~340 lines of duplicate code eliminated

### Documentation
- 📖 [CHANGES-LOG.md](./CHANGES-LOG.md) - Detailed changelog
- 📖 [OPTIMIZATION-SUMMARY.md](./OPTIMIZATION-SUMMARY.md) - Optimization results
- 📖 [API-MIGRATION-GUIDE.md](./API-MIGRATION-GUIDE.md) - Frontend migration guide
- 📖 [ARCHITECTURE-OPTIMIZED.md](./ARCHITECTURE-OPTIMIZED.md) - Updated architecture

## 🚀 Quick Start

### Prerequisites
- Java 17+
- Python 3.9+
- Redis
- PostgreSQL
- Neo4j

### 1. Start Python Service

```bash
cd python-model

# Install dependencies
pip install -r requirements.txt

# Start service
python server.py
```

Python service will start on **http://localhost:8000**

### 2. Start Java Service

```bash
cd java-api

# Build
./mvnw clean package

# Run
./mvnw spring-boot:run
```

Java service will start on **http://localhost:8081**

### 3. Verify Services

```bash
# Check Python service
curl http://localhost:8000/health

# Check Java service
curl http://localhost:8081/api/recommendations/health
```

## 📡 API Endpoints

### Main Endpoints

#### Get Personalized Feed
```http
GET /api/recommendations/feed?userId={id}&page=0&size=20

Response:
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
  "generatedAt": "2024-12-08T10:00:00"
}
```

#### Record User Interaction
```http
POST /api/recommendations/interaction

Body:
{
  "userId": "123",
  "postId": "p1",
  "type": "LIKE",
  "viewDuration": 5.2,
  "context": {}
}

Response:
{
  "status": "success",
  "message": "Interaction recorded"
}
```

#### Refresh Cache
```http
POST /api/recommendations/refresh?userId={id}

Response:
{
  "status": "success",
  "message": "Cache refreshed for user: 123"
}
```

Full API documentation: [API-MIGRATION-GUIDE.md](./API-MIGRATION-GUIDE.md)

## 🧪 Testing

### Manual Testing

```bash
# 1. Test Python service
curl http://localhost:8000/health

# 2. Test embedding generation
curl -X POST http://localhost:8000/embed/post \
  -H "Content-Type: application/json" \
  -d '{
    "post_id": "test1",
    "content": "Mạng máy tính chương 4",
    "title": "TCP/IP"
  }'

# 3. Test Java service
curl http://localhost:8081/api/recommendations/health

# 4. Test feed generation
curl "http://localhost:8081/api/recommendations/feed?userId=test123&size=10"
```

### Integration Testing

See [test-recommendation-api.ps1](./test-recommendation-api.ps1) for automated tests.

## 📂 Project Structure

```
recommend-service/
├── java-api/                    # Java Spring Boot Service
│   ├── src/main/java/vn/ctu/edu/recommend/
│   │   ├── controller/
│   │   │   └── RecommendationController.java    (✅ UNIFIED)
│   │   ├── service/
│   │   │   └── HybridRecommendationService.java (✅ MAIN)
│   │   ├── client/
│   │   │   ├── PythonModelServiceClient.java
│   │   │   ├── UserServiceClient.java (Feign)
│   │   │   └── PostServiceClient.java (Feign)
│   │   ├── kafka/
│   │   ├── repository/
│   │   ├── model/
│   │   └── config/
│   └── pom.xml
│
├── python-model/                # Python AI Service
│   ├── server.py               (✅ UNIFIED ENTRY POINT)
│   ├── inference.py            (PhoBERT Engine)
│   ├── api/
│   │   └── routes.py           (ML Routes)
│   ├── services/
│   │   └── prediction_service.py
│   ├── models/
│   │   └── schemas.py
│   ├── utils/
│   └── requirements.txt
│
├── ARCHITECTURE.md             # Original architecture
├── ARCHITECTURE-OPTIMIZED.md   # Optimized architecture
├── CHANGES-LOG.md              # Detailed changes
├── OPTIMIZATION-SUMMARY.md     # Optimization summary
├── API-MIGRATION-GUIDE.md      # Migration guide
└── README-OPTIMIZED.md         # This file
```

## 🔧 Configuration

### Java Service (application.yml)

```yaml
recommendation:
  python-service:
    enabled: true
    url: http://localhost:8000
  cache:
    min-ttl: 30    # seconds
    max-ttl: 120   # seconds
  default-recommendation-count: 20
```

### Python Service (config.py)

```python
MODEL_PATH = "model/academic_posts_model"
PHOBERT_MODEL_NAME = "vinai/phobert-base"
EMBEDDING_DIMENSION = 768
PORT = 8000
```

## 📊 Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Get Feed (cached) | 5-10ms | Redis cache hit |
| Get Feed (uncached) | 150-300ms | Full ML pipeline |
| Generate Embedding | 50-100ms | PhoBERT inference |
| Record Interaction | 2-5ms | Async Kafka event |

## 🔄 Scaling

### Horizontal Scaling

```bash
# Scale Java service
docker-compose up --scale java-api=3

# Scale Python service  
docker-compose up --scale python-model=3
```

### Resource Requirements

- **Java Service**: 2-4 GB RAM, 2 CPUs
- **Python Service**: 4-8 GB RAM, 2-4 CPUs (PhoBERT model)
- **Redis**: 2-4 GB RAM
- **PostgreSQL**: 4-8 GB RAM

## 🐛 Troubleshooting

### Python service not responding
```bash
# Check if Python service is running
curl http://localhost:8000/health

# Check logs
tail -f python-model/logs/python-service-*.log
```

### Java can't connect to Python
```bash
# Verify Python service URL in application.yml
recommendation.python-service.url=http://localhost:8000

# Check network connectivity
curl http://localhost:8000/health
```

### Cache not working
```bash
# Check Redis connection
redis-cli ping

# Check Redis config in application.yml
spring.redis.host=localhost
spring.redis.port=6379
```

## 📚 Additional Resources

### Documentation
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Original architecture document
- [README-RECOMMENDATION-SERVICE.md](../README-RECOMMENDATION-SERVICE.md) - Overall system README

### Related Services
- **user-service**: User profile and authentication
- **post-service**: Post management and timeline
- **api-gateway**: API routing and load balancing

## 🤝 Contributing

When making changes:
1. Follow the architecture in ARCHITECTURE-OPTIMIZED.md
2. Keep Java = Orchestrator, Python = AI only
3. Don't duplicate endpoints or logic
4. Update documentation
5. Add tests

## 📝 License

© 2024 CTU Connect. All rights reserved.

---

## 🔖 Version History

### v2.0.0 (2024-12-08) - Optimization Release
- ✅ Consolidated controllers (2 → 1)
- ✅ Unified Python entry point (2 → 1)
- ✅ Single API pattern
- ✅ Removed ~340 lines of duplicate code
- ✅ Enhanced documentation

### v1.0.0 (2024-11-XX) - Initial Release
- Initial hybrid architecture
- PhoBERT integration
- Basic recommendation engine

---

**For detailed changes, see** [CHANGES-LOG.md](./CHANGES-LOG.md)
