# 🚀 HƯỚNG DẪN SETUP VÀ PHÁT TRIỂN RECOMMENDATION SERVICE

## 📋 MỤC LỤC
1. [Tổng quan](#tổng-quan)
2. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
3. [Cài đặt môi trường](#cài-đặt-môi-trường)
4. [Chạy services trên IDE](#chạy-services-trên-ide)
5. [Kiểm tra kết nối](#kiểm-tra-kết-nối)
6. [Test hệ thống](#test-hệ-thống)
7. [Tích hợp với services khác](#tích-hợp-với-services-khác)
8. [Troubleshooting](#troubleshooting)

---

## 📊 TỔNG QUAN

### Recommendation Service gồm 2 phần:

#### 1️⃣ **Java Service (recommendation-service-java)** - Port 8095
- **Vai trò:** API Gateway, Business Logic, Database Operations
- **Công nghệ:** Spring Boot 3, PostgreSQL, Neo4j, Redis, Kafka
- **Chức năng:**
  - Nhận request từ client/API Gateway
  - Lấy user profile và candidate posts từ database
  - Gọi Python service để ranking bằng ML
  - Apply business rules và cache results
  - Trả về recommendations cho user

#### 2️⃣ **Python Service (recommendation-service-python)** - Port 8097
- **Vai trò:** Machine Learning Engine, NLP Processing
- **Công nghệ:** FastAPI, PyTorch, PhoBERT, scikit-learn
- **Chức năng:**
  - Nhận request từ Java service
  - Generate text embeddings bằng PhoBERT
  - Tính toán content similarity
  - ML-based ranking và scoring
  - Trả về ranked results cho Java service

### 🔄 Luồng hoạt động:
```
User Request → API Gateway → Java Service → Python ML Service
                                    ↓              ↓
                               PostgreSQL     PhoBERT Model
                               Neo4j          Similarity
                               Redis          Ranking
                                    ↓              ↓
                                    ← Ranked Results ←
                                    ↓
                            Apply Business Rules
                                    ↓
                            Return to User
```

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

```
recommendation-service-java/          # Java Spring Boot Service
├── src/main/java/vn/ctu/edu/recommend/
│   ├── controller/                   # REST Controllers
│   │   └── RecommendationController.java
│   ├── service/                      # Business Logic
│   │   ├── HybridRecommendationService.java  # Core service
│   │   ├── UserProfileService.java
│   │   └── CandidatePostService.java
│   ├── client/                       # External Service Clients
│   │   └── PythonModelServiceClient.java     # Gọi Python service
│   ├── repository/                   # Data Access
│   ├── model/                        # Entities & DTOs
│   └── config/                       # Configuration
├── src/main/resources/
│   ├── application.yml               # Main config
│   └── application-dev.yml           # Dev config
└── pom.xml                           # Maven dependencies

recommendation-service-python/        # Python ML Service
├── app.py                           # FastAPI main application
├── config.py                        # Configuration
├── requirements.txt                 # Python dependencies
├── api/
│   └── routes.py                    # API endpoints
├── services/
│   ├── prediction_service.py        # Core ML logic
│   ├── embedding_service.py         # PhoBERT embeddings
│   └── ranking_service.py           # Ranking algorithms
├── models/
│   └── schemas.py                   # Request/Response models
└── utils/
    └── similarity.py                # Similarity calculations
```

---

## 🔧 CÀI ĐẶT MÔI TRƯỜNG

### ✅ Prerequisites

#### 1. Java Development Environment
- **JDK 17** hoặc cao hơn
- **Maven 3.8+**
- **IntelliJ IDEA** hoặc Eclipse (khuyến nghị IntelliJ)

#### 2. Python Development Environment
- **Python 3.10+**
- **pip** (Python package manager)
- **PyCharm** hoặc VS Code (khuyến nghị PyCharm)

#### 3. Databases & Services (Chạy bằng Docker)
- Docker Desktop installed
- PostgreSQL (port 5435)
- Neo4j (port 7687, 7474)
- Redis (port 6379)
- Kafka (port 9092)

---

## 🚀 CHẠY SERVICES TRÊN IDE

### BƯỚC 1: Start Databases (Docker)

Trước tiên, chạy các databases cần thiết bằng Docker Compose:

```powershell
# Navigate to Java service directory
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java

# Start databases only
docker-compose -f docker-compose.dev.yml up -d

# Verify containers are running
docker ps
```

**Kết quả mong đợi:**
```
CONTAINER ID   IMAGE                    PORTS                    STATUS
xxxxx          postgres:15-alpine       5435->5432               Up
xxxxx          neo4j:5.13.0            7474->7474, 7687->7687   Up
xxxxx          redis:7-alpine          6379->6379               Up
xxxxx          apache/kafka:3.7.0      9092->9092               Up
```

### BƯỚC 2: Setup Python Service

#### 2.1. Tạo Virtual Environment

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-python

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify Python version
python --version  # Should be 3.10+
```

#### 2.2. Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | Select-String -Pattern "fastapi|torch|transformers"
```

**Lưu ý:** Việc cài đặt PyTorch và Transformers có thể mất vài phút.

#### 2.3. Create .env File

Tạo file `.env` trong thư mục `recommendation-service-python/`:

```env
# Python ML Service Configuration
PORT=8097
DEBUG=True
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=./academic_posts_model
PHOBERT_MODEL=vinai/phobert-base
MAX_LENGTH=256
EMBEDDING_DIM=768

# Redis Configuration (Connect to Docker Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=1
REDIS_PASSWORD=

# Kafka Configuration (Connect to Docker Kafka)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREDICTIONS=recommendation.predictions

# Performance Settings
BATCH_SIZE=32
MAX_WORKERS=4
CACHE_TTL=3600
```

#### 2.4. Chạy Python Service

**Option A: Chạy trực tiếp từ Terminal**

```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Run the service
python app.py
```

**Option B: Chạy từ PyCharm**

1. Open `recommendation-service-python` folder in PyCharm
2. Configure Python Interpreter:
   - File → Settings → Project → Python Interpreter
   - Select the venv you created (`.\venv\Scripts\python.exe`)
3. Right-click `app.py` → Run 'app'

**Kết quả mong đợi:**

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8097 (Press CTRL+C to quit)
```

#### 2.5. Kiểm tra Python Service

Mở browser hoặc dùng curl:

```powershell
# Health check
curl http://localhost:8097/health

# API documentation (Swagger UI)
Start-Process http://localhost:8097/docs

# Test endpoint
curl http://localhost:8097/
```

**Response mong đợi:**
```json
{
  "service": "CTU Connect Recommendation ML Service",
  "version": "1.0.0",
  "status": "running",
  "timestamp": "2024-12-07T12:46:29.123456"
}
```

---

### BƯỚC 3: Setup Java Service

#### 3.1. Import Project vào IntelliJ IDEA

1. Open IntelliJ IDEA
2. File → Open → Select `recommendation-service-java` folder
3. IntelliJ sẽ tự động detect Maven project và import dependencies
4. Wait for Maven to download all dependencies (có thể mất vài phút)

#### 3.2. Configure Application Profile

Tạo/Edit file `application-dev.yml`:

```yaml
spring:
  datasource:
    url: jdbc:postgresql://localhost:5435/recommendation_db
    username: postgres
    password: postgres
  
  neo4j:
    uri: bolt://localhost:7687
    authentication:
      username: neo4j
      password: password
  
  data:
    redis:
      host: localhost
      port: 6379
  
  kafka:
    bootstrap-servers: localhost:9092

# Python Service Configuration
recommendation:
  python-service:
    url: http://localhost:8097
    enabled: true
    fallback-to-legacy: true
    timeout: 10000

# Server Configuration
server:
  port: 8095

# Eureka (Optional for dev)
eureka:
  client:
    enabled: false  # Disable Eureka in dev mode
```

#### 3.3. Chạy Java Service

**Option A: Run từ IntelliJ**

1. Mở `src/main/java/vn/ctu/edu/recommend/RecommendationServiceApplication.java`
2. Click vào nút ▶️ (Run) bên cạnh `public class RecommendationServiceApplication`
3. Hoặc Right-click → Run 'RecommendationServiceApplication'
4. Trong Run Configuration, thêm VM options nếu cần:
   ```
   -Dspring.profiles.active=dev
   ```

**Option B: Run bằng Maven**

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java

# Run with dev profile
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

**Kết quả mong đợi:**

```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::                (v3.3.4)

2024-12-07 19:46:30 INFO  RecommendationServiceApplication - Starting RecommendationServiceApplication
2024-12-07 19:46:31 INFO  RecommendationServiceApplication - No active profile set, falling back to default
2024-12-07 19:46:32 INFO  TomcatWebServer - Tomcat initialized with port(s): 8095 (http)
2024-12-07 19:46:33 INFO  RecommendationServiceApplication - Started RecommendationServiceApplication in 3.456 seconds
```

#### 3.4. Kiểm tra Java Service

```powershell
# Health check
curl http://localhost:8095/actuator/health

# Service info
curl http://localhost:8095/actuator/info
```

**Response mong đợi:**
```json
{
  "status": "UP",
  "components": {
    "db": {"status": "UP"},
    "redis": {"status": "UP"},
    "neo4j": {"status": "UP"},
    "python-service": {"status": "UP"}
  }
}
```

---

## 🔗 KIỂM TRA KẾT NỐI GIỮA 2 SERVICES

### Test 1: Java gọi Python trực tiếp

Từ IntelliJ console hoặc logs, bạn sẽ thấy:

```
INFO  PythonModelServiceClient - Checking Python service health...
INFO  PythonModelServiceClient - Python service health: UP
INFO  PythonModelServiceClient - Python service URL: http://localhost:8097
```

### Test 2: Manual API Call

```powershell
# Test Python service trực tiếp
curl -X POST http://localhost:8097/api/model/predict `
  -H "Content-Type: application/json" `
  -d '{
    "userAcademic": {
      "userId": "user123",
      "major": "CNTT",
      "interests": ["AI", "Machine Learning"]
    },
    "candidatePosts": [],
    "topK": 10
  }'
```

### Test 3: Test qua Java Service

```powershell
# Get recommendations through Java service
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=10"
```

---

## 🧪 TEST HỆ THỐNG

Tôi đã tạo script test tự động cho bạn. Xem file: `test-recommendation-dev.ps1`

### Chạy Full Test Suite

```powershell
cd d:\LVTN\CTU-Connect-demo

# Run test script
.\test-recommendation-dev.ps1
```

Script này sẽ test:
1. ✅ Database connectivity (PostgreSQL, Neo4j, Redis, Kafka)
2. ✅ Python service health
3. ✅ Java service health
4. ✅ Python ML endpoints
5. ✅ Java recommendation endpoints
6. ✅ Integration test (Java → Python)

---

## 🔌 TÍCH HỢP VỚI SERVICES KHÁC

### 1. Tích hợp với API Gateway

**Cấu hình trong API Gateway:**

File: `api-gateway/src/main/resources/application.yml`

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: recommendation-service
          uri: lb://recommendation-service  # Load balanced
          predicates:
            - Path=/api/recommendation/**
          filters:
            - StripPrefix=0
```

**Test qua API Gateway:**

```powershell
# Thay vì gọi trực tiếp port 8095, gọi qua API Gateway
curl http://localhost:8090/api/recommendation/feed?userId=user123&size=10
```

### 2. Tích hợp với User Service

Java Recommendation Service đã có code để lấy user profile từ Neo4j:

```java
// File: UserProfileService.java
public UserProfile getUserProfile(String userId) {
    // Lấy từ Neo4j graph database
    return neo4jRepository.findUserProfile(userId);
}
```

### 3. Tích hợp với Post Service

Java Recommendation Service đã có code để lấy posts từ MongoDB:

```java
// File: CandidatePostService.java
public List<Post> getCandidatePosts(UserProfile profile) {
    // Lấy từ MongoDB qua Feign client hoặc direct query
    return postRepository.findCandidatePosts(profile);
}
```

### 4. Kafka Events

Recommendation service lắng nghe các events:

**Topics được lắng nghe:**
- `user.interaction` - User likes, comments, shares
- `post.created` - New posts
- `user.profile.updated` - User profile changes

**File:** `kafka/KafkaConsumerConfig.java`

```java
@KafkaListener(topics = "user.interaction", groupId = "recommendation-group")
public void handleUserInteraction(UserInteractionEvent event) {
    // Update user interaction history
    // Invalidate cache if needed
}
```

---

## 🐛 TROUBLESHOOTING

### Vấn đề 1: Python service không start

**Triệu chứng:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Giải pháp:**
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

---

### Vấn đề 2: Java không kết nối được Python

**Triệu chứng:**
```
Connection refused: http://localhost:8097
```

**Giải pháp:**
1. Kiểm tra Python service đang chạy:
   ```powershell
   curl http://localhost:8097/health
   ```
2. Kiểm tra firewall
3. Kiểm tra URL trong `application-dev.yml`:
   ```yaml
   recommendation:
     python-service:
       url: http://localhost:8097  # Đảm bảo đúng port
   ```

---

### Vấn đề 3: Database connection failed

**Triệu chứng:**
```
Connection to localhost:5435 refused
```

**Giải pháp:**
```powershell
# Check Docker containers
docker ps

# If not running, start them
cd recommendation-service-java
docker-compose -f docker-compose.dev.yml up -d

# Check logs
docker logs recommendation-postgres
```

---

### Vấn đề 4: Out of Memory khi chạy Python

**Triệu chứng:**
```
torch.cuda.OutOfMemoryError
```

**Giải pháp:**

Edit `config.py`:
```python
# Use CPU instead of GPU for development
DEVICE = "cpu"
BATCH_SIZE = 8  # Reduce batch size
```

---

### Vấn đề 5: Maven build failed

**Triệu chứng:**
```
Failed to execute goal
```

**Giải pháp:**
```powershell
# Clean and rebuild
mvn clean install -DskipTests

# If still fails, delete .m2 cache
Remove-Item -Recurse -Force "$env:USERPROFILE\.m2\repository"
mvn clean install
```

---

## 📊 MONITORING & DEBUGGING

### 1. Check Logs

**Python Service:**
```powershell
# Logs in terminal where Python is running
# Or check log files
Get-Content recommendation-service-python\logs\*.log -Tail 50
```

**Java Service:**
```powershell
# IntelliJ console
# Or check log files if configured
Get-Content recommendation-service-java\logs\*.log -Tail 50
```

### 2. Redis Cache Monitoring

```powershell
# Connect to Redis
docker exec -it recommendation-redis redis-cli

# Check cached keys
KEYS recommendation:*

# Check specific key
GET recommendation:feed:user123

# Flush cache if needed
FLUSHDB
```

### 3. Database Queries

**PostgreSQL:**
```powershell
docker exec -it recommendation-postgres psql -U postgres -d recommendation_db

# Check tables
\dt

# Sample query
SELECT * FROM user_interactions LIMIT 10;
```

**Neo4j:**
```powershell
# Open browser
Start-Process http://localhost:7474

# Sample Cypher query
MATCH (u:User {userId: 'user123'})-[r]->(n)
RETURN u, r, n LIMIT 25;
```

---

## 📈 NEXT STEPS

Sau khi setup xong và test thành công:

1. ✅ Thêm test data vào database
2. ✅ Train ML models với data thực
3. ✅ Fine-tune scoring weights
4. ✅ Optimize cache strategy
5. ✅ Setup monitoring (Prometheus, Grafana)
6. ✅ Deploy to Docker (sau khi phát triển xong)

---

## 📞 CẦN TRỢ GIÚP?

### Quick Checks:

```powershell
# 1. Check all services
docker ps
curl http://localhost:8097/health  # Python
curl http://localhost:8095/actuator/health  # Java

# 2. Run test script
.\test-recommendation-dev.ps1

# 3. Check logs
# Python: Terminal output
# Java: IntelliJ console
```

### Common Commands:

```powershell
# Restart Python service
# Ctrl+C in terminal, then:
python app.py

# Restart Java service
# Stop in IntelliJ, then Run again

# Restart databases
docker-compose -f docker-compose.dev.yml restart

# Clear cache
docker exec recommendation-redis redis-cli FLUSHDB
```

---

**🎉 HOÀN TẤT!** Bạn đã setup xong Recommendation Service và sẵn sàng phát triển!

**📝 Nhớ:** 
- Python service (port 8097) phải chạy trước
- Java service (port 8095) sẽ tự động kết nối đến Python
- Databases phải chạy trong Docker
- Check logs thường xuyên để debug

**🚀 Happy Coding!**
