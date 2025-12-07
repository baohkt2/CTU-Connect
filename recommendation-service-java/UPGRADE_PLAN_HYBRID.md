# 🚀 UPGRADE PLAN: Recommendation Service Hybrid Architecture

## 📋 Tổng quan

Nâng cấp recommendation-service-java theo kiến trúc Hybrid ML hoàn chỉnh, kết hợp Python ML layer với Java Business layer để đạt hiệu năng và độ chính xác cao.

---

## ✅ Trạng thái hiện tại

### Đã có (Implemented):
1. ✅ Java Spring Boot service với cấu trúc hoàn chỉnh
2. ✅ Integration với PostgreSQL, Neo4j, Redis
3. ✅ Kafka producer/consumer infrastructure
4. ✅ PythonModelServiceClient để gọi Python service
5. ✅ HybridRecommendationService với caching
6. ✅ UserServiceClient để lấy thông tin user
7. ✅ Redis caching với TTL adaptive
8. ✅ Kafka event streaming cho user interactions
9. ✅ Business rules và filtering logic
10. ✅ API endpoints hoàn chỉnh

### Cần cải thiện:
1. ⚠️ Python Model Service chưa có (cần tạo mới)
2. ⚠️ Training pipeline chưa hoàn thiện
3. ⚠️ Chưa có model files (academic_posts_model/)
4. ⚠️ Integration với post-service/user-service
5. ⚠️ Testing data pipeline

---

## 🎯 Các bước triển khai

### PHASE 1: Hoàn thiện Java Service (CURRENT) ✅

**Mục tiêu:** Đảm bảo Java service hoạt động ổn định

#### 1.1. Kiểm tra và fix các service hiện tại
```bash
# Kiểm tra service đang chạy
curl http://localhost:8095/actuator/health

# Test feed endpoint
curl http://localhost:8095/api/recommendation/feed?userId=user123
```

#### 1.2. Đảm bảo database schema đúng
- PostgreSQL: PostEmbedding, UserFeedback tables
- Neo4j: User, Post nodes và relationships
- Redis: Cache keys structure

#### 1.3. Verify Kafka topics
```bash
# List topics
kafka-topics.sh --list --bootstrap-server localhost:9092

# Create if missing
kafka-topics.sh --create --topic user_interaction --bootstrap-server localhost:9092
kafka-topics.sh --create --topic post_viewed --bootstrap-server localhost:9092
kafka-topics.sh --create --topic post_liked --bootstrap-server localhost:9092
kafka-topics.sh --create --topic recommendation_training_data --bootstrap-server localhost:9092
```

---

### PHASE 2: Tạo Python Model Service 🔧

**Mục tiêu:** Xây dựng Python FastAPI service cho ML operations

#### 2.1. Cấu trúc thư mục
```
recommendation-service/
├── app.py                          # FastAPI main
├── requirements.txt                # Dependencies
├── models/
│   ├── embedding_model.py          # PhoBERT embedding
│   ├── academic_classifier.py      # Academic content classifier
│   └── ranking_model.py            # ML ranking model
├── academic_posts_model/           # Pre-trained models
│   ├── vectorizer.pkl
│   ├── post_encoder.pkl
│   ├── academic_encoder.pkl
│   └── ranking_model.pkl
├── api/
│   └── routes.py                   # API endpoints
├── services/
│   ├── prediction_service.py       # Main prediction logic
│   └── feature_service.py          # Feature engineering
├── utils/
│   └── similarity.py               # Similarity calculations
└── config.py                       # Configuration
```

#### 2.2. Core API Endpoint
```python
# POST /api/model/predict
{
  "userAcademic": {...},
  "userHistory": [...],
  "candidatePosts": [...],
  "topK": 20
}
```

#### 2.3. Dependencies
```txt
fastapi==0.104.1
uvicorn==0.24.0
transformers==4.35.0
torch==2.1.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.3
pydantic==2.5.0
redis==5.0.1
```

---

### PHASE 3: Training Pipeline & Model 🤖

**Mục tiêu:** Setup training pipeline và pre-trained models

#### 3.1. Datasets Structure
```json
// academic_dataset.json
{
  "userProfile": {
    "major": "Computer Science",
    "faculty": "Engineering",
    "degree": "Bachelor",
    "batch": "K48"
  },
  "post": {
    "content": "Hội thảo Machine Learning...",
    "hashtags": ["#AI", "#Workshop"],
    "mediaDescription": "Poster event",
    "authorMajor": "Computer Science",
    "authorFaculty": "Engineering"
  },
  "interaction": {
    "liked": 1,
    "commented": 0,
    "shared": 0,
    "viewDuration": 4.3
  },
  "timestamp": 1234567890
}
```

#### 3.2. Kafka Consumer for Training
```python
# training/kafka_consumer.py
- Read from Kafka topics
- Append to datasets
- Trigger retraining
```

#### 3.3. Training Script
```python
# training/train_model.py
- Load datasets
- Train PhoBERT embeddings
- Train ranking model
- Export to pkl files
- Deploy to Python service
```

---

### PHASE 4: Integration Testing 🧪

**Mục tiêu:** Test end-to-end flow

#### 4.1. Unit Tests
```bash
# Java tests
mvn test

# Python tests
pytest tests/
```

#### 4.2. Integration Tests
```bash
# Test complete flow
./test-hybrid-api.ps1
```

#### 4.3. Load Testing
```bash
# JMeter or k6
k6 run load-test.js
```

---

### PHASE 5: Deployment & Monitoring 🚀

**Mục tiêu:** Deploy và monitor hệ thống

#### 5.1. Docker Compose Update
```yaml
services:
  recommendation-java:
    build: ./recommendation-service-java
    environment:
      PYTHON_MODEL_SERVICE_URL: http://recommendation-python:8097
    
  recommendation-python:
    build: ./recommendation-service
    ports:
      - "8097:8097"
    volumes:
      - ./academic_posts_model:/app/models
```

#### 5.2. Monitoring
- Prometheus metrics
- Grafana dashboards
- Log aggregation (ELK)
- Performance tracking

---

## 📊 Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Java Service | 1 day | ✅ DONE |
| Phase 2: Python Service | 3 days | 🔧 TODO |
| Phase 3: Training Pipeline | 2 days | 🔧 TODO |
| Phase 4: Testing | 2 days | 🔧 TODO |
| Phase 5: Deployment | 1 day | 🔧 TODO |

**Total:** 9 days

---

## 🔧 Quick Start Commands

### Start Databases Only
```bash
cd recommendation-service-java
docker-compose -f docker-compose.dev.yml up postgres neo4j redis kafka
```

### Run Java Service in IDE
```
1. Open IntelliJ IDEA
2. Import Maven project
3. Configure Run Configuration:
   - Main class: RecommendationServiceApplication
   - VM options: -Dspring.profiles.active=dev
   - Environment: See .env.example
4. Run/Debug
```

### Run Python Service (After Phase 2)
```bash
cd recommendation-service
pip install -r requirements.txt
python app.py
```

### Test API
```bash
# Java service health
curl http://localhost:8095/actuator/health

# Get recommendations
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=20"

# Check Python service (after Phase 2)
curl http://localhost:8097/health
```

---

## 📝 Configuration Checklist

### Java Service (.env or environment variables)
```properties
POSTGRES_HOST=localhost
POSTGRES_PORT=5435
NEO4J_HOST=localhost
NEO4J_PORT=7687
REDIS_HOST=localhost
REDIS_PORT=6379
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
PYTHON_MODEL_SERVICE_URL=http://localhost:8097
EUREKA_SERVER_URL=http://localhost:8761/eureka/
```

### Python Service (.env)
```properties
MODEL_PATH=./academic_posts_model
REDIS_HOST=localhost
REDIS_PORT=6379
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
PORT=8097
```

---

## 🐛 Troubleshooting

### Java Service không start
```bash
# Check databases
docker ps | grep postgres
docker ps | grep neo4j
docker ps | grep redis

# Check logs
tail -f logs/recommendation-service.log
```

### Python Service không connect
```bash
# Check if Python service running
curl http://localhost:8097/health

# Java fallback mode
# Java service sẽ dùng popularity-based ranking nếu Python service down
```

### Cache issues
```bash
# Clear Redis cache
redis-cli FLUSHDB

# Check cache keys
redis-cli KEYS recommend:*
```

---

## 📚 Next Steps

1. ✅ **Phase 1 Done** - Java service hoàn chỉnh
2. 🔧 **Phase 2** - Tạo Python Model Service
   - Create FastAPI structure
   - Implement prediction endpoint
   - Load pre-trained models
3. 🔧 **Phase 3** - Setup Training Pipeline
   - Kafka consumer
   - Dataset management
   - Model retraining logic
4. 🔧 **Phase 4** - Testing
   - Unit tests
   - Integration tests
   - Performance tests
5. 🔧 **Phase 5** - Deployment
   - Docker images
   - CI/CD pipeline
   - Monitoring setup

---

## 💡 Best Practices

1. **Always use fallback** - Java service có fallback khi Python service down
2. **Cache aggressively** - Redis cache giảm load lên Python service
3. **Monitor latency** - Track response times via Prometheus
4. **Incremental training** - Update model thường xuyên từ user feedback
5. **A/B testing** - Test new models trước khi deploy production

---

## 📞 Support

Gặp vấn đề? Check:
1. `HYBRID_ARCHITECTURE.md` - Kiến trúc chi tiết
2. `TESTING_GUIDE.md` - Hướng dẫn test
3. `logs/` - Application logs
4. `docker-compose.dev.yml` - Database configs

---

**Status:** ✅ Java Service Ready | 🔧 Python Service Pending
