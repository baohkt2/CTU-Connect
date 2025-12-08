# 🎓 CTU Connect - Recommendation Service

Hệ thống gợi ý nội dung học thuật thông minh sử dụng PhoBERT và kiến trúc hybrid Python-Java.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc](#kiến-trúc)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [API Documentation](#api-documentation)
- [Testing](#testing)

---

## 🎯 Tổng quan

Recommendation Service là hệ thống gợi ý nội dung học thuật cá nhân hóa cho mạng xã hội CTU Connect. Hệ thống kết hợp:

* **PhoBERT Model** - Xử lý ngôn ngữ tiếng Việt
* **Python FastAPI** - AI Inference Engine
* **Java Spring Boot** - API Gateway & Orchestration
* **Redis** - Embedding Cache
* **Kafka** - Real-time Event Processing

### Tính năng chính

✅ Gợi ý bài viết học thuật phù hợp với chuyên ngành  
✅ Gợi ý bạn bè theo sở thích học tập  
✅ Gợi ý tài liệu, học liệu liên quan  
✅ Personalized News Feed  
✅ Real-time embedding generation  
✅ Scalable & High Performance  

---

## 🏗 Kiến trúc

### Component Diagram

```
┌─────────────────────────────────────┐
│         Frontend (React)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Java API (Spring Boot)         │
│  - REST Controllers                 │
│  - Recommendation Logic             │
│  - Ranking Algorithm                │
│  - Cache Management                 │
└──────┬──────────────────────┬───────┘
       │                      │
       ▼                      ▼
┌──────────────────┐   ┌─────────────────┐
│  Python Service  │   │   Data Layer    │
│  - PhoBERT       │   │  - PostgreSQL   │
│  - Embedding Gen │   │  - Neo4j        │
│  - Similarity    │   │  - Redis        │
└──────────────────┘   └─────────────────┘
```

### Service Communication

* **Java → Python**: HTTP REST API
* **Java → Redis**: Spring Data Redis
* **Java → Kafka**: Spring Kafka
* **Java → PostgreSQL**: Spring Data JPA
* **Java → Neo4j**: Spring Data Neo4j

---

## 🚀 Cài đặt

### Prerequisites

* **Java 17+**
* **Python 3.10+**
* **Maven 3.8+**
* **Docker & Docker Compose**
* **Redis**
* **Kafka**
* **PostgreSQL**
* **Neo4j**

### 1. Clone Repository

```bash
cd recommend-service
```

### 2. Setup Python Service

```bash
cd python-model

# Create virtual environment
python -m venv venv

# Activate venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download or place PhoBERT model in model/academic_posts_model/
```

### 3. Setup Java Service

```bash
cd java-api

# Build with Maven
./mvnw clean install

# Or with Maven wrapper
mvn clean install
```

### 4. Docker Setup (Recommended)

```bash
cd docker
docker-compose up -d
```

---

## ⚙️ Configuration

### Python Service (.env)

```env
MODEL_PATH=./model/academic_posts_model
LOG_LEVEL=INFO
WORKERS=2
PORT=8000
```

### Java Service (application.yml)

```yaml
server:
  port: 8081

spring:
  application:
    name: recommendation-service
    
  # Python Service Configuration
  python:
    inference:
      url: http://localhost:8000
      
  # Redis Configuration
  redis:
    host: localhost
    port: 6379
    
  # Kafka Configuration
  kafka:
    bootstrap-servers: localhost:9092
    consumer:
      group-id: recommendation-service
      
  # PostgreSQL Configuration
  datasource:
    url: jdbc:postgresql://localhost:5432/ctuconnect
    username: postgres
    password: postgres
    
  # Neo4j Configuration
  data:
    neo4j:
      uri: bolt://localhost:7687
      username: neo4j
      password: password
```

---

## 🎮 Sử dụng

### Start Services

#### Option 1: Docker (Recommended)

```bash
cd docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Option 2: Manual Start

**Terminal 1 - Python Service:**
```bash
cd python-model
uvicorn server:app --reload --port 8000
```

**Terminal 2 - Java Service:**
```bash
cd java-api
./mvnw spring-boot:run
```

### Verify Services

```bash
# Check Python service
curl http://localhost:8000/health

# Check Java service
curl http://localhost:8081/actuator/health
```

---

## 📚 API Documentation

### Python Inference API (Port 8000)

#### Generate Post Embedding

```http
POST /embed/post
Content-Type: application/json

{
  "post_id": "post123",
  "content": "Mạng máy tính - giao thức TCP/IP",
  "title": "Bài giảng TCP/IP"
}

Response:
{
  "id": "post123",
  "embedding": [0.123, -0.456, ...],
  "dimension": 768
}
```

#### Generate User Embedding

```http
POST /embed/user
Content-Type: application/json

{
  "user_id": "user123",
  "major": "Khoa học máy tính",
  "faculty": "Công nghệ thông tin",
  "courses": ["Mạng máy tính", "Cơ sở dữ liệu"],
  "skills": ["Python", "Java"],
  "bio": "Sinh viên năm 3 CNTT"
}

Response:
{
  "id": "user123",
  "embedding": [0.789, -0.234, ...],
  "dimension": 768
}
```

#### Compute Similarity

```http
POST /similarity/batch
Content-Type: application/json

{
  "query_embedding": [0.1, 0.2, ...],
  "candidate_embeddings": [
    [0.3, 0.4, ...],
    [0.5, 0.6, ...]
  ]
}

Response:
{
  "similarities": [0.85, 0.72],
  "count": 2
}
```

### Java Recommendation API (Port 8081)

#### Get Personalized Feed

```http
GET /api/recommendations/feed?userId=user123&page=0&size=20

Response:
{
  "posts": [
    {
      "postId": "post1",
      "title": "TCP/IP Protocol",
      "content": "...",
      "score": 0.95,
      "reason": "Phù hợp với chuyên ngành của bạn"
    }
  ],
  "total": 50,
  "page": 0,
  "size": 20
}
```

#### Get Academic Recommendations

```http
GET /api/recommendations/academic?userId=user123&subject=network

Response:
{
  "posts": [...]
}
```

#### Refresh User Embeddings

```http
POST /api/recommendations/refresh
Content-Type: application/json

{
  "userId": "user123"
}

Response:
{
  "status": "success",
  "message": "User embedding refreshed"
}
```

---

## 🧪 Testing

### Test Python Service

```bash
cd python-model
pytest tests/
```

### Test Java Service

```bash
cd java-api
./mvnw test
```

### Integration Tests

```bash
# Test full recommendation flow
curl -X GET "http://localhost:8081/api/recommendations/feed?userId=test-user"
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8081/api/recommendations/feed?userId=test
```

---

## 📊 Monitoring

### Metrics Endpoints

* Python Service: `http://localhost:8000/health`
* Java Service: `http://localhost:8081/actuator/health`
* Java Metrics: `http://localhost:8081/actuator/metrics`

### Logging

Logs are stored in:
* Python: `python-model/logs/`
* Java: `java-api/logs/`

---

## 🔧 Troubleshooting

### Python Service Issues

**Problem:** Model not loading
```bash
# Check model path
ls python-model/model/academic_posts_model/

# Should contain:
# - pytorch_model.bin
# - config.json
# - tokenizer/
```

**Problem:** Out of memory
```bash
# Reduce batch size in config
BATCH_SIZE=16
```

### Java Service Issues

**Problem:** Cannot connect to Python service
```bash
# Check Python service is running
curl http://localhost:8000/health

# Check configuration
spring.python.inference.url=http://localhost:8000
```

**Problem:** Redis connection failed
```bash
# Check Redis is running
redis-cli ping

# Should return PONG
```

---

## 📖 Documentation

* [Full Architecture](./ARCHITECTURE.md) - Kiến trúc chi tiết
* [API Reference](./API.md) - API Documentation
* [Development Guide](./DEVELOPMENT.md) - Hướng dẫn phát triển

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👥 Team

CTU Connect Development Team

---

## 🙏 Acknowledgments

* PhoBERT by VinAI Research
* Spring Boot Framework
* FastAPI Framework
* Hugging Face Transformers
