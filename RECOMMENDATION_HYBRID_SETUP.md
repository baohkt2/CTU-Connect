# 🚀 CTU Connect Recommendation System - Hybrid Architecture Setup Guide

## 📋 Tổng quan

Hệ thống recommendation sử dụng **Hybrid Architecture** kết hợp:
- **Python ML Service** - Machine Learning, NLP, Embedding (Port 8097)
- **Java Spring Boot Service** - API, Business Logic, Orchestration (Port 8095)
- **Redis** - Caching (Port 6379)
- **PostgreSQL** - Metadata storage (Port 5435)
- **Neo4j** - Graph relationships (Port 7687)
- **Kafka** - Event streaming (Port 9092)

---

## ✅ Trạng thái hiện tại

### Đã hoàn thành:
1. ✅ **Java Service** (`recommendation-service-java/`)
   - Spring Boot application hoàn chỉnh
   - Integration với DB (PostgreSQL, Neo4j, Redis)
   - Kafka producer/consumer
   - Business logic và filtering
   - Caching layer
   
2. ✅ **Python ML Service** (`recommendation-service-python/`)
   - FastAPI structure
   - PhoBERT embedding integration
   - Prediction API endpoint
   - Feature engineering utilities
   - Hot reload capability

### Cần làm tiếp:
1. 🔧 **Training Pipeline** - Kafka consumer và model retraining
2. 🔧 **Pre-trained Models** - Train initial models
3. 🔧 **Integration Testing** - End-to-end testing
4. 🔧 **Monitoring** - Prometheus + Grafana dashboards

---

## 🎯 Quick Start - Development Mode

### Bước 1: Khởi động Databases

```powershell
# Di chuyển vào thư mục Java service
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java

# Khởi động các database
docker-compose -f docker-compose.dev.yml up -d postgres neo4j redis kafka
```

Kiểm tra services đã chạy:
```powershell
docker ps
```

Expected output: 4 containers running (postgres, neo4j, redis, kafka)

---

### Bước 2: Khởi động Java Service (IntelliJ IDEA)

#### Option A: Run trong IntelliJ

1. **Open Project**
   ```
   File -> Open -> d:\LVTN\CTU-Connect-demo\recommendation-service-java
   ```

2. **Wait for Maven import** (bottom right progress bar)

3. **Configure Run Configuration**
   - Click "Add Configuration" (top right)
   - Add new "Application"
   - Main class: `vn.ctu.edu.recommend.RecommendationServiceApplication`
   - VM options: `-Dspring.profiles.active=dev`
   - Environment variables:
     ```
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5435
     NEO4J_HOST=localhost
     NEO4J_PORT=7687
     REDIS_HOST=localhost
     REDIS_PORT=6379
     KAFKA_BOOTSTRAP_SERVERS=localhost:9092
     PYTHON_MODEL_SERVICE_URL=http://localhost:8097
     ```

4. **Run** (Shift+F10 or Green play button)

#### Option B: Maven Command Line

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

#### Verify Java Service

```powershell
# Health check
curl http://localhost:8095/actuator/health

# Expected: {"status":"UP"}
```

---

### Bước 3: Khởi động Python ML Service

#### Tạo virtual environment

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-python

# Tạo virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Note:** Bước này sẽ mất 5-10 phút để download PyTorch và Transformers.

#### Cấu hình environment

```powershell
# Copy config file
Copy-Item .env.example .env

# Edit .env nếu cần (hoặc dùng default)
```

#### Run Python service

```powershell
# Development mode
python app.py
```

Service sẽ chạy trên port 8097.

#### Verify Python Service

```powershell
# Health check
curl http://localhost:8097/health

# API docs
# Open browser: http://localhost:8097/docs
```

---

### Bước 4: Test End-to-End

#### Test Java service without Python

```powershell
# Get recommendations (will use fallback if Python not ready)
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=10"
```

#### Test Python service directly

```powershell
# Test prediction endpoint
curl -X POST http://localhost:8097/api/model/predict `
  -H "Content-Type: application/json" `
  -d '{
    "userAcademic": {
      "userId": "user123",
      "major": "Computer Science",
      "faculty": "Engineering",
      "degree": "Bachelor",
      "batch": "K48"
    },
    "userHistory": [],
    "candidatePosts": [
      {
        "postId": "post1",
        "content": "Hội thảo Machine Learning tại CTU",
        "hashtags": ["#AI", "#Workshop"],
        "authorMajor": "Computer Science",
        "authorFaculty": "Engineering",
        "likesCount": 10,
        "commentsCount": 5,
        "sharesCount": 2
      }
    ],
    "topK": 5
  }'
```

#### Test full integration

```powershell
# Java service sẽ gọi Python service
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=10"

# Check logs trong IntelliJ để thấy:
# - "Calling Python model service..."
# - "Python model returned X ranked posts"
```

---

## 📊 Architecture Flow

```
User Request
    ↓
Java Service (8095)
    ├─→ Check Redis Cache (Fast path)
    ├─→ Get User Profile
    ├─→ Get Candidate Posts
    ├─→ Call Python ML Service (8097)
    │       ├─→ Generate Embeddings (PhoBERT)
    │       ├─→ Calculate Similarity
    │       ├─→ Calculate Academic Score
    │       ├─→ Rank Posts
    │       └─→ Return Ranked List
    ├─→ Apply Business Rules
    ├─→ Cache Results
    └─→ Return to User
```

---

## 🔧 Development Workflow

### Khi phát triển Java code

1. Edit code in IntelliJ
2. Restart service (Ctrl+F5 hoặc click Rerun)
3. Test API

### Khi phát triển Python code

```powershell
# Python service auto-reloads in dev mode
# Just save file, service will restart automatically
```

### Load test data

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java

# Load PostgreSQL test data
docker exec -i recommendation-postgres psql -U postgres -d recommendation_db < database\init.sql

# Load Neo4j test data
docker exec -i recommendation-neo4j cypher-shell -u neo4j -p password < test-data.cypher
```

---

## 🐛 Troubleshooting

### Java service không start

**Lỗi: Cannot connect to database**
```powershell
# Check databases running
docker ps | Select-String "postgres|neo4j|redis"

# Check ports
netstat -an | Select-String "5435|7687|6379"

# Restart databases
docker-compose -f docker-compose.dev.yml restart
```

**Lỗi: Port 8095 already in use**
```powershell
# Find process using port
netstat -ano | Select-String "8095"

# Kill process (replace <PID>)
taskkill /PID <PID> /F
```

### Python service không start

**Lỗi: ModuleNotFoundError**
```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

**Lỗi: torch not found hoặc CUDA error**
```powershell
# Uninstall và reinstall torch cho CPU
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Lỗi: Port 8097 already in use**
```powershell
# Find and kill process
netstat -ano | Select-String "8097"
taskkill /PID <PID> /F
```

### Java không gọi được Python

**Check Python service health:**
```powershell
curl http://localhost:8097/health
```

**Check Java configuration:**
```properties
# application-dev.yml
recommendation:
  python-service:
    url: http://localhost:8097  # Phải đúng
    enabled: true                # Phải là true
```

**Check logs:**
```
# Java logs (IntelliJ Console)
# Look for: "Calling Python model service..."
# If error: "Python service unavailable, using fallback"

# Python logs (Terminal)
# Look for: "POST /api/model/predict"
```

---

## 📝 Configuration Files

### Java: `application-dev.yml`

```yaml
recommendation:
  python-service:
    url: http://localhost:8097
    enabled: true
    fallback-to-legacy: true
  
  cache:
    recommendation-ttl: 120
    min-ttl: 30
    max-ttl: 120
  
  weights:
    content-similarity: 0.35
    graph-relation: 0.30
    academic-score: 0.25
    popularity-score: 0.10
```

### Python: `.env`

```properties
PORT=8097
DEBUG=true
MODEL_PATH=./academic_posts_model
REDIS_HOST=localhost
REDIS_PORT=6379
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

---

## 🧪 Testing

### Unit Tests (Java)

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java
mvn test
```

### Unit Tests (Python)

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-python
pytest tests/ -v
```

### Integration Test

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java
.\test-hybrid-api.ps1
```

---

## 📈 Performance Monitoring

### Metrics Endpoints

- **Java:** http://localhost:8095/actuator/metrics
- **Python:** http://localhost:8097/metrics
- **Prometheus:** http://localhost:8095/actuator/prometheus

### Check Cache Performance

```powershell
# Redis stats
docker exec -it recommendation-redis redis-cli INFO stats

# Check cache keys
docker exec -it recommendation-redis redis-cli KEYS "recommend:*"
```

### Check Kafka Topics

```powershell
# List topics
docker exec -it recommendation-kafka kafka-topics --list --bootstrap-server localhost:9092

# Check consumer group
docker exec -it recommendation-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group recommendation-service-group
```

---

## 🔄 Next Steps

### Phase 1: ✅ Basic Setup (DONE)
- Java service running
- Python service running
- Basic API working

### Phase 2: 🔧 Model Training (TODO)
1. Collect training data
2. Train initial models
3. Save model files to `academic_posts_model/`
4. Test with real models

### Phase 3: 🔧 Training Pipeline (TODO)
1. Implement Kafka consumer
2. Dataset management
3. Incremental training
4. Model versioning

### Phase 4: 🔧 Production Deployment (TODO)
1. Docker Compose for all services
2. CI/CD pipeline
3. Monitoring setup
4. Load testing

---

## 💡 Tips

1. **Always start databases first** - Java service cần DB để khởi động
2. **Use fallback mode** - Java service hoạt động ngay cả khi Python service down
3. **Monitor logs** - IntelliJ console + Python terminal
4. **Clear cache** - Khi test, clear Redis cache để thấy kết quả mới
5. **Hot reload** - Python service auto-reload khi save file

---

## 📚 Documentation

- **Architecture:** `recommendation-service-java/HYBRID_ARCHITECTURE.md`
- **Java Docs:** `recommendation-service-java/README.md`
- **Python Docs:** `recommendation-service-python/README.md`
- **Testing:** `recommendation-service-java/TESTING_GUIDE.md`
- **Upgrade Plan:** `recommendation-service-java/UPGRADE_PLAN_HYBRID.md`

---

## 📞 Need Help?

### Check Logs
```powershell
# Java logs
# IntelliJ Console hoặc logs/recommendation-service.log

# Python logs
# Terminal output hoặc logs/python-service-*.log

# Database logs
docker logs recommendation-postgres
docker logs recommendation-neo4j
docker logs recommendation-redis
```

### Health Checks
```powershell
# All services
curl http://localhost:8095/actuator/health  # Java
curl http://localhost:8097/health           # Python
docker ps                                    # Databases
```

---

**Status:** ✅ Ready for Development | 🔧 Training Pipeline Pending

**Next:** Tạo training data và train model ban đầu
