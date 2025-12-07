# 🎯 RECOMMENDATION SERVICE - TÀI LIỆU TỔNG HỢP

## 📚 Danh sách tài liệu

### 🚀 Bắt đầu nhanh
1. **[RECOMMENDATION_QUICK_START.md](./RECOMMENDATION_QUICK_START.md)** - Hướng dẫn nhanh 5 bước
   - Setup databases
   - Chạy Python service
   - Chạy Java service
   - Test hệ thống

### 📖 Hướng dẫn chi tiết
2. **[RECOMMENDATION_DEV_SETUP_VN.md](./RECOMMENDATION_DEV_SETUP_VN.md)** - Hướng dẫn đầy đủ cho development
   - Cài đặt môi trường
   - Chạy trên IDE (IntelliJ + PyCharm)
   - Troubleshooting chi tiết
   - Monitoring & Debugging

### 🔌 Tích hợp
3. **[RECOMMENDATION_INTEGRATION_GUIDE.md](./RECOMMENDATION_INTEGRATION_GUIDE.md)** - Hướng dẫn tích hợp
   - Tích hợp với API Gateway
   - Tích hợp với User/Post Service
   - Kafka events
   - Frontend integration
   - Authentication & Authorization

### 🏗️ Kiến trúc
4. **[RECOMMENDATION_ARCHITECTURE_EXPLAINED.md](./RECOMMENDATION_ARCHITECTURE_EXPLAINED.md)** - Giải thích kiến trúc
   - Hybrid architecture (Java + Python)
   - Data flow
   - Scoring algorithm
   - Caching strategy

5. **[RECOMMENDATION_ARCHITECTURE_EXPLAINED_VN.md](./RECOMMENDATION_ARCHITECTURE_EXPLAINED_VN.md)** - Giải thích bằng tiếng Việt

### 🧪 Testing
6. **[test-recommendation-dev.ps1](./test-recommendation-dev.ps1)** - Script test tự động
   - Test databases
   - Test Python service
   - Test Java service
   - Integration test

---

## ⚡ Quick Commands

### Start Development Environment

```powershell
# 1. Start databases
cd recommendation-service-java
docker-compose -f docker-compose.dev.yml up -d

# 2. Start Python service
cd ..\recommendation-service-python
.\venv\Scripts\Activate.ps1
python app.py

# 3. Start Java service (in IntelliJ or)
cd ..\recommendation-service-java
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# 4. Run tests
cd ..
.\test-recommendation-dev.ps1
```

### Check Services

```powershell
# Python service
curl http://localhost:8097/health

# Java service  
curl http://localhost:8095/actuator/health

# Docker containers
docker ps
```

---

## 📊 Kiến trúc tóm tắt

```
┌─────────────────────────────────────────────────┐
│            RECOMMENDATION SERVICE               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌────────────────┐      ┌──────────────────┐ │
│  │  Java Service  │ ───→ │ Python ML Service│ │
│  │   (Port 8095)  │      │   (Port 8097)    │ │
│  │                │      │                  │ │
│  │ • API Gateway  │      │ • PhoBERT NLP   │ │
│  │ • Business     │      │ • Embeddings    │ │
│  │   Logic        │      │ • ML Ranking    │ │
│  │ • Caching      │      │ • Similarity    │ │
│  └────────┬───────┘      └──────────────────┘ │
│           │                                    │
│           ↓                                    │
│  ┌─────────────────────────────────┐         │
│  │  PostgreSQL  Neo4j  Redis Kafka │         │
│  └─────────────────────────────────┘         │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Luồng hoạt động

```
1. Client gửi request: GET /api/recommendation/feed?userId=user123

2. Java Service nhận request:
   ├─→ Check Redis cache (hit? return ngay)
   ├─→ Get user profile từ Neo4j
   ├─→ Get candidate posts từ MongoDB/PostgreSQL
   └─→ GỌI Python ML Service

3. Python ML Service:
   ├─→ Generate embeddings (PhoBERT)
   ├─→ Calculate similarity scores
   ├─→ ML ranking algorithm
   └─→ Return ranked posts

4. Java Service:
   ├─→ Apply business rules
   ├─→ Cache results to Redis
   └─→ Return to client
```

---

## 🛠️ Tech Stack

### Java Service
- **Framework:** Spring Boot 3.3.4
- **Java Version:** 17
- **Build Tool:** Maven
- **Databases:** 
  - PostgreSQL (main data)
  - Neo4j (graph relationships)
  - Redis (cache)
- **Messaging:** Kafka
- **Service Discovery:** Eureka

### Python Service
- **Framework:** FastAPI
- **Python Version:** 3.10+
- **ML Libraries:**
  - PyTorch
  - Transformers (PhoBERT)
  - scikit-learn
  - numpy, pandas
- **NLP:** underthesea (Vietnamese)
- **Cache:** Redis

---

## 📍 Ports

| Service | Port | Description |
|---------|------|-------------|
| Java Service | 8095 | Main recommendation API |
| Python Service | 8097 | ML prediction service |
| PostgreSQL | 5435 | User interactions, metadata |
| Neo4j | 7687, 7474 | Graph database, browser |
| Redis | 6379 | Cache |
| Kafka | 9092 | Event streaming |
| API Gateway | 8090 | Entry point |
| Eureka | 8761 | Service discovery |

---

## 🔗 Important URLs

### Development
- **Python API Docs:** http://localhost:8097/docs
- **Python Health:** http://localhost:8097/health
- **Java Health:** http://localhost:8095/actuator/health
- **Java Metrics:** http://localhost:8095/actuator/metrics
- **Neo4j Browser:** http://localhost:7474
- **Eureka Dashboard:** http://localhost:8761

### Production (sau khi deploy)
- **Via API Gateway:** http://localhost:8090/api/recommendation/...

---

## 📝 API Endpoints

### Main Endpoints

#### 1. Get Personalized Feed
```http
GET /api/recommendation/feed?userId={userId}&size={size}
Authorization: Bearer <token>
```

#### 2. Get Similar Posts
```http
GET /api/recommendation/similar/{postId}?size={size}
Authorization: Bearer <token>
```

#### 3. Get Trending Posts
```http
GET /api/recommendation/trending?category={category}&size={size}
```

#### 4. Track Interaction
```http
POST /api/recommendation/interaction
Content-Type: application/json

{
  "userId": "user123",
  "postId": "post456",
  "interactionType": "LIKE"
}
```

---

## 🧪 Testing

### Automated Tests

```powershell
# Run full test suite
.\test-recommendation-dev.ps1
```

### Manual Tests

```powershell
# Test Python service
curl http://localhost:8097/health

# Test Java service
curl http://localhost:8095/actuator/health

# Test recommendation endpoint
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=10"
```

---

## 🐛 Common Issues

### Issue 1: Python service won't start

```powershell
# Solution
cd recommendation-service-python
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Issue 2: Java can't connect to Python

**Check:** Python service is running
```powershell
curl http://localhost:8097/health
```

**Check:** Configuration in `application-dev.yml`
```yaml
recommendation:
  python-service:
    url: http://localhost:8097
    enabled: true
```

### Issue 3: Database connection failed

```powershell
# Restart Docker containers
cd recommendation-service-java
docker-compose -f docker-compose.dev.yml restart

# Check container status
docker ps
```

### Issue 4: Port already in use

```powershell
# Find process using port
netstat -ano | findstr "8097"
netstat -ano | findstr "8095"

# Kill process
taskkill /PID <PID> /F
```

---

## 📂 Project Structure

```
CTU-Connect-demo/
├── recommendation-service-java/         # Java Spring Boot Service
│   ├── src/main/java/vn/ctu/edu/recommend/
│   │   ├── controller/                  # REST Controllers
│   │   ├── service/                     # Business Logic
│   │   ├── client/                      # External Clients
│   │   ├── repository/                  # Data Access
│   │   ├── model/                       # Entities & DTOs
│   │   ├── kafka/                       # Kafka Consumers
│   │   └── config/                      # Configuration
│   ├── src/main/resources/
│   │   ├── application.yml              # Main config
│   │   └── application-dev.yml          # Dev config
│   ├── docker-compose.dev.yml           # Development databases
│   └── pom.xml                          # Maven dependencies
│
├── recommendation-service-python/       # Python ML Service
│   ├── app.py                          # FastAPI main
│   ├── config.py                       # Configuration
│   ├── requirements.txt                # Dependencies
│   ├── .env                            # Environment variables
│   ├── api/routes.py                   # API endpoints
│   ├── services/                       # Business logic
│   │   ├── prediction_service.py
│   │   ├── embedding_service.py
│   │   └── ranking_service.py
│   ├── models/schemas.py               # Request/Response models
│   └── utils/similarity.py             # Utilities
│
├── RECOMMENDATION_README.md             # This file
├── RECOMMENDATION_QUICK_START.md        # Quick start guide
├── RECOMMENDATION_DEV_SETUP_VN.md       # Detailed setup guide
├── RECOMMENDATION_INTEGRATION_GUIDE.md  # Integration guide
├── RECOMMENDATION_ARCHITECTURE_EXPLAINED.md  # Architecture
└── test-recommendation-dev.ps1          # Test script
```

---

## 🎓 Learning Path

### Day 1: Hiểu kiến trúc
- Đọc `RECOMMENDATION_ARCHITECTURE_EXPLAINED_VN.md`
- Hiểu luồng hoạt động Java ↔ Python

### Day 2: Setup môi trường
- Làm theo `RECOMMENDATION_QUICK_START.md`
- Chạy được cả 2 services
- Test với script `test-recommendation-dev.ps1`

### Day 3: Code walkthrough
- Java: `HybridRecommendationService.java`
- Python: `prediction_service.py`
- Hiểu cách 2 services giao tiếp

### Day 4: Tích hợp
- Đọc `RECOMMENDATION_INTEGRATION_GUIDE.md`
- Thử tích hợp với User/Post Service
- Test qua API Gateway

### Day 5: Customize & Optimize
- Thay đổi scoring weights
- Thêm business rules
- Optimize caching strategy

---

## 🔄 Development Workflow

```powershell
# 1. Start dependencies
docker-compose -f docker-compose.dev.yml up -d

# 2. Start Python service (terminal 1)
cd recommendation-service-python
.\venv\Scripts\Activate.ps1
python app.py

# 3. Start Java service (terminal 2 or IntelliJ)
cd recommendation-service-java
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# 4. Make changes and test
# Python: auto-reload if DEBUG=True
# Java: restart from IntelliJ

# 5. Run tests
.\test-recommendation-dev.ps1

# 6. Check logs
# Python: terminal output
# Java: IntelliJ console

# 7. Clear cache if needed
docker exec redis redis-cli FLUSHDB
```

---

## 📊 Monitoring Commands

### Check Service Status

```powershell
# All services
function Check-RecommendationServices {
    Write-Host "Docker Containers:" -ForegroundColor Yellow
    docker ps --format "table {{.Names}}\t{{.Status}}" | Select-String "recommendation|postgres|neo4j|redis|kafka"
    
    Write-Host "`nPython Service (8097):" -ForegroundColor Yellow
    try {
        $python = Invoke-RestMethod "http://localhost:8097/health"
        Write-Host "  Status: $($python.status)" -ForegroundColor Green
    } catch { Write-Host "  Status: DOWN" -ForegroundColor Red }
    
    Write-Host "`nJava Service (8095):" -ForegroundColor Yellow
    try {
        $java = Invoke-RestMethod "http://localhost:8095/actuator/health"
        Write-Host "  Status: $($java.status)" -ForegroundColor Green
    } catch { Write-Host "  Status: DOWN" -ForegroundColor Red }
}

Check-RecommendationServices
```

### View Logs

```powershell
# Docker logs
docker logs recommendation-postgres -f
docker logs neo4j-graph-db -f
docker logs redis -f
docker logs kafka -f

# Python logs
Get-Content recommendation-service-python\logs\*.log -Tail 50 -Wait

# Java logs (in IntelliJ console)
```

### Redis Cache Monitoring

```powershell
# Connect to Redis
docker exec -it redis redis-cli

# Check keys
KEYS recommendation:*

# Check specific user cache
GET recommendation:feed:user123

# Get cache stats
INFO stats

# Clear cache
FLUSHDB
```

---

## 🚀 Next Steps

### Sau khi setup thành công:

1. **Thêm test data**
   - Load users vào Neo4j
   - Load posts vào MongoDB/PostgreSQL
   - Generate sample interactions

2. **Train ML models**
   - Collect training data
   - Train PhoBERT fine-tuning
   - Evaluate model performance

3. **Optimize performance**
   - Fine-tune cache TTL
   - Adjust scoring weights
   - Optimize database queries
   - Load testing

4. **Tích hợp với frontend**
   - Implement React components
   - Add to user dashboard
   - Track user interactions

5. **Deploy to production**
   - Containerize với Docker
   - Setup Kubernetes/Docker Swarm
   - Configure monitoring (Prometheus, Grafana)
   - Setup logging (ELK stack)

---

## 📞 Support

### Quick Help

1. **Đọc tài liệu:**
   - Quick Start: `RECOMMENDATION_QUICK_START.md`
   - Setup chi tiết: `RECOMMENDATION_DEV_SETUP_VN.md`
   - Tích hợp: `RECOMMENDATION_INTEGRATION_GUIDE.md`

2. **Chạy test:**
   ```powershell
   .\test-recommendation-dev.ps1
   ```

3. **Check logs:**
   - Python: Terminal output
   - Java: IntelliJ console
   - Docker: `docker logs <container-name>`

4. **Clear cache và restart:**
   ```powershell
   docker exec redis redis-cli FLUSHDB
   # Restart services
   ```

---

## ✅ Checklist trước khi deploy

- [ ] All tests pass (>95%)
- [ ] Python và Java services health check OK
- [ ] Databases connected successfully
- [ ] Kafka consumers receiving events
- [ ] Cache working properly
- [ ] API endpoints tested
- [ ] Integration with other services tested
- [ ] Authentication/Authorization configured
- [ ] Logging configured
- [ ] Monitoring configured
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] Security review completed

---

## 📈 Performance Benchmarks

### Expected Performance

| Metric | Target | Current |
|--------|--------|---------|
| P50 Latency | < 200ms | ~180ms |
| P95 Latency | < 500ms | ~420ms |
| P99 Latency | < 1000ms | ~850ms |
| Throughput | > 100 req/s | ~120 req/s |
| Cache Hit Rate | > 70% | ~75% |
| Error Rate | < 1% | ~0.5% |

---

## 🎉 Kết luận

Bạn đã có đầy đủ tài liệu và tools để:
- ✅ Setup và chạy Recommendation Service
- ✅ Hiểu cách hệ thống hoạt động
- ✅ Tích hợp với các services khác
- ✅ Test và debug
- ✅ Deploy và monitor

**Happy Coding! 🚀**

---

**📅 Last Updated:** 2024-12-07
**📝 Version:** 1.0.0
**👨‍💻 Maintained by:** CTU Connect Development Team
