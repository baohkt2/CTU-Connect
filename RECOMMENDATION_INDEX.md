# 📚 CTU Connect Recommendation System - Documentation Index

## 🎯 Start Here

Bạn đang tìm gì?

### 🚀 Tôi muốn khởi động hệ thống
→ **[RECOMMENDATION_HYBRID_SETUP.md](RECOMMENDATION_HYBRID_SETUP.md)**
- Complete setup guide từ A-Z
- Khởi động Java + Python services
- Test end-to-end
- Troubleshooting common issues

### 📊 Tôi muốn hiểu kiến trúc
→ **[recommendation-service-java/HYBRID_ARCHITECTURE.md](recommendation-service-java/HYBRID_ARCHITECTURE.md)**
- Kiến trúc 3 layers chi tiết
- Data flow diagrams
- Component responsibilities
- Performance metrics

### ✅ Tôi muốn biết đã làm được gì
→ **[RECOMMENDATION_IMPLEMENTATION_SUMMARY.md](RECOMMENDATION_IMPLEMENTATION_SUMMARY.md)**
- Current status
- What works
- What needs work
- Next steps roadmap

### 🔧 Tôi muốn cải tiến/nâng cấp
→ **[recommendation-service-java/UPGRADE_PLAN_HYBRID.md](recommendation-service-java/UPGRADE_PLAN_HYBRID.md)**
- Upgrade phases
- Timeline estimates
- Task breakdown
- Configuration guides

---

## 📂 Project Structure

```
CTU-Connect-demo/
├── recommendation-service-java/          # Java Spring Boot Service
│   ├── src/main/java/                   # Java source code
│   │   └── vn/ctu/edu/recommend/
│   │       ├── client/                  # External service clients
│   │       ├── config/                  # Configuration
│   │       ├── controller/              # REST controllers
│   │       ├── service/                 # Business logic
│   │       ├── kafka/                   # Kafka producers/consumers
│   │       ├── model/                   # Entities & DTOs
│   │       └── repository/              # Data access
│   ├── src/main/resources/              # Configuration files
│   │   ├── application.yml              # Base config
│   │   ├── application-dev.yml          # Development config
│   │   └── application-docker.yml       # Docker config
│   ├── docker-compose.dev.yml           # Development databases
│   ├── pom.xml                          # Maven dependencies
│   ├── HYBRID_ARCHITECTURE.md           # Architecture docs
│   ├── UPGRADE_PLAN_HYBRID.md           # Upgrade guide
│   └── README.md                        # Java service docs
│
├── recommendation-service-python/        # Python ML Service
│   ├── app.py                           # FastAPI main
│   ├── config.py                        # Configuration
│   ├── requirements.txt                 # Python dependencies
│   ├── api/                             # API routes
│   │   └── routes.py                    # Endpoints
│   ├── models/                          # Pydantic models
│   │   └── schemas.py                   # Request/Response models
│   ├── services/                        # Business logic
│   │   └── prediction_service.py        # ML predictions
│   ├── utils/                           # Utilities
│   │   ├── similarity.py                # Similarity calculations
│   │   └── feature_engineering.py       # Feature extraction
│   ├── training/                        # Training pipeline
│   ├── academic_posts_model/            # Pre-trained models
│   ├── Dockerfile                       # Docker image
│   └── README.md                        # Python service docs
│
├── RECOMMENDATION_HYBRID_SETUP.md        # 🔥 MAIN SETUP GUIDE
├── RECOMMENDATION_IMPLEMENTATION_SUMMARY.md  # Current status
├── RECOMMENDATION_INDEX.md               # This file
└── test-hybrid-recommendation.ps1        # Test script
```

---

## 🔑 Key Files

### Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `application.yml` | Java base config | `recommendation-service-java/src/main/resources/` |
| `application-dev.yml` | Java dev config | `recommendation-service-java/src/main/resources/` |
| `config.py` | Python config | `recommendation-service-python/` |
| `.env` | Python environment | `recommendation-service-python/` |
| `docker-compose.dev.yml` | Development databases | `recommendation-service-java/` |
| `pom.xml` | Java dependencies | `recommendation-service-java/` |
| `requirements.txt` | Python dependencies | `recommendation-service-python/` |

### Documentation Files

| File | Purpose |
|------|---------|
| `RECOMMENDATION_HYBRID_SETUP.md` | Complete setup guide |
| `RECOMMENDATION_IMPLEMENTATION_SUMMARY.md` | Implementation status |
| `RECOMMENDATION_INDEX.md` | This documentation index |
| `recommendation-service-java/HYBRID_ARCHITECTURE.md` | Architecture details |
| `recommendation-service-java/UPGRADE_PLAN_HYBRID.md` | Upgrade roadmap |
| `recommendation-service-java/README.md` | Java service documentation |
| `recommendation-service-python/README.md` | Python service documentation |

### Scripts

| Script | Purpose |
|--------|---------|
| `test-hybrid-recommendation.ps1` | Test both Java and Python services |
| `recommendation-service-java/start-dev.ps1` | Start Java service |
| `recommendation-service-java/test-api.ps1` | Test Java API |
| `recommendation-service-java/load-test-data.ps1` | Load test data |

---

## 🎯 Common Tasks

### Task 1: Khởi động hệ thống lần đầu

```powershell
# 1. Read setup guide
Get-Content RECOMMENDATION_HYBRID_SETUP.md

# 2. Start databases
cd recommendation-service-java
docker-compose -f docker-compose.dev.yml up -d

# 3. Start Java service (IntelliJ or Maven)
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# 4. Start Python service
cd ..\recommendation-service-python
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py

# 5. Test
cd ..
.\test-hybrid-recommendation.ps1
```

### Task 2: Debug issues

```powershell
# Check service health
curl http://localhost:8095/actuator/health  # Java
curl http://localhost:8097/health           # Python

# Check databases
docker ps

# View logs
# Java: IntelliJ console
# Python: Terminal output
# Docker: docker logs <container-name>

# Clear cache
docker exec -it recommendation-redis redis-cli FLUSHDB
```

### Task 3: Test API

```powershell
# Run test script
.\test-hybrid-recommendation.ps1

# Manual tests
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=10"
curl http://localhost:8097/api/model/info
```

### Task 4: Deploy changes

```powershell
# Java changes
cd recommendation-service-java
mvn clean package
# Restart service

# Python changes
cd recommendation-service-python
# Save file - auto reloads in dev mode

# Database changes
docker exec -i recommendation-postgres psql -U postgres -d recommendation_db < schema.sql
```

---

## 📊 Architecture Quick Reference

### Services & Ports

| Service | Port | Technology | Status |
|---------|------|------------|--------|
| Java Recommendation Service | 8095 | Spring Boot 3 | ✅ Running |
| Python ML Service | 8097 | FastAPI | ✅ Running |
| PostgreSQL | 5435 | Database | ✅ Running |
| Neo4j | 7687, 7474 | Graph DB | ✅ Running |
| Redis | 6379 | Cache | ✅ Running |
| Kafka | 9092 | Event Stream | ✅ Running |

### Data Flow

```
User → Java Service → Check Cache
                    ↓ (miss)
                    Get User Profile
                    Get Candidate Posts
                    ↓
                    Python ML Service
                    ├─ PhoBERT Embedding
                    ├─ Similarity Calculation
                    ├─ Academic Scoring
                    └─ Ranking
                    ↓
                    Apply Business Rules
                    Cache Results
                    ↓
                    Return to User
```

### Scoring Formula

```python
final_score = (
    0.35 * content_similarity +    # PhoBERT cosine similarity
    0.30 * implicit_feedback +     # User interaction history
    0.25 * academic_score +        # Academic relevance
    0.10 * popularity_score        # Engagement metrics
)
```

---

## 🧪 Testing

### Quick Tests

```powershell
# All tests
.\test-hybrid-recommendation.ps1

# Java only
cd recommendation-service-java
mvn test

# Python only
cd recommendation-service-python
pytest tests/ -v
```

### Manual API Tests

```powershell
# Java health
curl http://localhost:8095/actuator/health

# Python health
curl http://localhost:8097/health

# Get recommendations
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=10"

# Python prediction
curl -X POST http://localhost:8097/api/model/predict `
  -H "Content-Type: application/json" `
  -d @test-request.json
```

---

## 🐛 Troubleshooting Index

### Java Service Issues

| Problem | Solution | Reference |
|---------|----------|-----------|
| Port 8095 in use | Kill process or change port | Setup Guide Section "Troubleshooting" |
| Database connection failed | Check Docker containers | `docker ps` |
| Maven build fails | Check Java version (need 17) | `java -version` |
| Service won't start | Check logs | IntelliJ Console |

### Python Service Issues

| Problem | Solution | Reference |
|---------|----------|-----------|
| Port 8097 in use | Kill process or change port | Setup Guide Section "Troubleshooting" |
| Module not found | Activate venv, reinstall | `pip install -r requirements.txt` |
| Torch/CUDA error | Install CPU version | See Setup Guide |
| Model not loading | Check MODEL_PATH | `.env` file |

### Integration Issues

| Problem | Solution | Reference |
|---------|----------|-----------|
| Java can't reach Python | Check Python service health | `curl http://localhost:8097/health` |
| No recommendations | Check fallback mode logs | Java logs |
| Cache not working | Check Redis connection | `docker exec recommendation-redis redis-cli PING` |
| Kafka not receiving events | Check Kafka topics | `docker exec recommendation-kafka kafka-topics --list` |

---

## 📚 Learning Path

### For New Developers

1. **Day 1:** Read architecture
   - `HYBRID_ARCHITECTURE.md`
   - `RECOMMENDATION_IMPLEMENTATION_SUMMARY.md`

2. **Day 2:** Setup environment
   - Follow `RECOMMENDATION_HYBRID_SETUP.md`
   - Run test script

3. **Day 3:** Understand code
   - Java: `HybridRecommendationService.java`
   - Python: `prediction_service.py`

4. **Day 4:** Make changes
   - Try modifying scoring weights
   - Test with different data

### For ML Engineers

Focus on:
- `recommendation-service-python/` - All ML code here
- `services/prediction_service.py` - Main prediction logic
- `utils/similarity.py` - Similarity calculations
- Training pipeline (TODO)

### For Backend Engineers

Focus on:
- `recommendation-service-java/` - All Java code here
- `service/HybridRecommendationService.java` - Main orchestration
- `client/PythonModelServiceClient.java` - Python integration
- `kafka/` - Event streaming

---

## 🔄 Update History

| Date | Version | Changes |
|------|---------|---------|
| 2024-12-07 | 1.0.0 | Initial hybrid architecture implementation |

---

## 📞 Need Help?

### Quick Checks

1. ✅ Read setup guide: `RECOMMENDATION_HYBRID_SETUP.md`
2. ✅ Check service health: `curl http://localhost:8095/actuator/health`
3. ✅ Run test script: `.\test-hybrid-recommendation.ps1`
4. ✅ Check logs: IntelliJ console / Python terminal
5. ✅ Check documentation: See links above

### Debug Commands

```powershell
# Check all services
docker ps
curl http://localhost:8095/actuator/health
curl http://localhost:8097/health

# Check logs
docker logs recommendation-postgres
docker logs recommendation-neo4j
docker logs recommendation-redis
docker logs recommendation-kafka

# Check cache
docker exec recommendation-redis redis-cli INFO stats

# Check Kafka topics
docker exec recommendation-kafka kafka-topics --list --bootstrap-server localhost:9092
```

---

**📌 Remember:** Start with `RECOMMENDATION_HYBRID_SETUP.md` for step-by-step setup!

**📌 Quick Test:** Run `.\test-hybrid-recommendation.ps1` to verify everything works!

**📌 Current Status:** ✅ Core Implementation Complete | 🔧 Training Pipeline Pending
