# 🎓 CTU-Connect

> **Mạng xã hội học thuật thông minh cho sinh viên và giảng viên Đại học Cần Thơ**

[![Java](https://img.shields.io/badge/Java-17-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)](https://www.java.com/)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.2.0-6DB33F?style=for-the-badge&logo=spring-boot)](https://spring.io/projects/spring-boot)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng chính](#-tính-năng-chính)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt và chạy](#-cài-đặt-và-chạy)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [API Documentation](#-api-documentation)
- [Database Schema](#-database-schema)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributors](#-contributors)

---

## 🎯 Giới thiệu

**CTU-Connect** là một nền tảng mạng xã hội học thuật được xây dựng riêng cho cộng đồng Đại học Cần Thơ, tích hợp công nghệ **AI/Machine Learning** để cung cấp trải nghiệm cá nhân hóa thông minh.

### Vấn đề giải quyết

- 📚 **Khó khăn trong chia sẻ tài liệu học tập**: Sinh viên gặp khó khăn khi tìm kiếm và chia sẻ tài liệu học thuật phù hợp với chuyên ngành
- 🤝 **Thiếu kết nối học thuật**: Khó tìm kiếm bạn bè và người hướng dẫn có cùng sở thích nghiên cứu
- 📰 **Information Overload**: Quá tải thông tin, khó lọc nội dung phù hợp với nhu cầu cá nhân
- 💬 **Giao tiếp phân tán**: Thiếu nền tảng tập trung để sinh viên và giảng viên trao đổi

### Giải pháp

CTU-Connect cung cấp một nền tảng tích hợp với các tính năng thông minh:

- 🤖 **AI-Powered Recommendation**: Gợi ý bài viết và bạn bè dựa trên PhoBERT (Vietnamese BERT)
- 🎓 **Academic Focus**: Tối ưu hóa cho nội dung học thuật, nghiên cứu khoa học
- 📊 **Personalization**: Cá nhân hóa feed dựa trên chuyên ngành, khoa, sở thích
- 💬 **Real-time Chat**: Trò chuyện real-time với WebSocket
- 🔍 **Smart Search**: Tìm kiếm nội dung và người dùng thông minh

---

## ✨ Tính năng chính

### 1. 📝 Quản lý bài viết học thuật
- Đăng bài viết với text, ảnh, hashtags
- Phân loại tự động nội dung học thuật/phi học thuật
- Like, Comment, Share
- Visibility control (Public, Friends, Private)

### 2. 📰 News Feed thông minh (AI-Powered)
- **Hybrid Recommendation Engine**:
  - Content-based filtering với PhoBERT embeddings (35%)
  - Implicit feedback từ lịch sử tương tác (25%)
  - Academic relevance (cùng ngành, khoa) (25%)
  - Popularity score (15%)
- Cache thông minh với Redis (TTL 30-120s)
- Infinite scroll với pagination

### 3. 👥 Gợi ý kết bạn (ML-Enhanced)
- **Multi-signal Friend Recommendation**:
  - PhoBERT similarity trên profile (30%)
  - Mutual friends (25%)
  - Academic connections (20%)
  - Activity score (15%)
  - Recency (10%)
- Lý do gợi ý rõ ràng (mutual friends, same major)
- Cache 6 giờ

### 4. 💬 Chat real-time
- WebSocket cho messaging real-time
- Typing indicators
- Read receipts (✓✓)
- Online/Offline presence
- Message persistence trong MongoDB
- Push notifications

### 5. 🔐 Xác thực và phân quyền
- JWT-based authentication
- Email verification (CTU domain only)
- Role-based access control (USER, ADMIN, LECTURER)
- OAuth2 integration ready

### 6. 🔍 Tìm kiếm và lọc
- Search users by name, email, student ID
- Filter by faculty, major, batch, gender
- Search posts by content, hashtags
- Advanced filters với Neo4j graph queries

---

## 🏗️ Kiến trúc hệ thống

### Tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER (React + Vite)                   │
│                     Port: 5173 (Client), 5174 (Admin)            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (Spring Cloud)                    │
│              Service Discovery + Routing + Load Balancing        │
│                          Port: 8090                              │
└─────┬────────────┬──────────┬──────────┬──────────┬────────────┘
      │            │          │          │          │
      ▼            ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│   Auth   │ │   User   │ │   Post   │ │   Chat   │ │  Media   │
│ Service  │ │ Service  │ │ Service  │ │ Service  │ │ Service  │
│  :8091   │ │  :8092   │ │  :8093   │ │  :8094   │ │  :8096   │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │            │
     ▼            ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│PostgreSQL│ │  Neo4j   │ │ MongoDB  │ │ MongoDB  │ │PostgreSQL│
│ auth_db  │ │ Graph DB │ │ post_db  │ │ chat_db  │ │ media_db │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘

┌─────────────────────────────────────────────────────────────────┐
│          RECOMMENDATION SERVICE (Hybrid: Java + Python)          │
│  ┌────────────────────────┐      ┌──────────────────────────┐  │
│  │   Java Orchestrator    │◄────►│   Python AI Engine       │  │
│  │   Port: 8095           │ HTTP │   Port: 8000             │  │
│  │  - API Gateway         │      │   - PhoBERT Model        │  │
│  │  - Business Logic      │      │   - Embedding Generation │  │
│  │  - Cache Management    │      │   - Similarity Compute   │  │
│  └───────┬────────────────┘      └──────────────────────────┘  │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────┐        ┌──────────────────┐             │
│  │   PostgreSQL     │        │      Redis       │             │
│  │   recommend_db   │        │   Cache Layer    │             │
│  └──────────────────┘        └──────────────────┘             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Kafka     │  │    Redis    │  │  Eureka     │             │
│  │ Event Bus   │  │ Cache/Session│  │  Discovery  │             │
│  │  :9092      │  │    :6379    │  │   :8761     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Kiến trúc Microservices

#### Core Services:
1. **Auth Service (8091)**: Authentication, JWT, Email verification
2. **User Service (8092)**: User profiles, relationships, friend management
3. **Post Service (8093)**: Posts, comments, likes, shares
4. **Chat Service (8094)**: Real-time messaging với WebSocket
5. **Media Service (8096)**: File upload, image processing
6. **Recommendation Service (8095)**: AI-powered recommendations

#### Infrastructure:
- **API Gateway (8090)**: Routing, load balancing, rate limiting
- **Eureka Server (8761)**: Service discovery
- **Kafka (9092)**: Event streaming
- **Redis (6379/6380)**: Caching, session management

---

## 🛠️ Công nghệ sử dụng

### Backend

#### Java Stack
- **Spring Boot 3.2.0** - Core framework
- **Spring Cloud Gateway** - API Gateway
- **Spring Cloud Netflix Eureka** - Service Discovery
- **Spring Data JPA** - Database ORM
- **Spring Data Neo4j** - Graph database
- **Spring Kafka** - Event streaming
- **Spring WebSocket** - Real-time communication
- **Spring Security** - Authentication & Authorization
- **JWT (JJWT)** - Token-based auth
- **OpenFeign** - Declarative REST client
- **Lombok** - Reduce boilerplate code

#### Python Stack
- **FastAPI** - High-performance web framework
- **PyTorch** - Deep learning framework
- **Transformers (HuggingFace)** - PhoBERT model
- **NumPy** - Numerical computing
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Frontend
- **React 18.2.0** - UI library
- **Vite** - Build tool
- **React Router** - Routing
- **Axios** - HTTP client
- **Socket.io Client** - WebSocket
- **TailwindCSS / Material-UI** - Styling

### Databases
- **PostgreSQL 14** - Relational data (Auth, Media, Recommend)
- **MongoDB 6.0** - Document store (Posts, Chat)
- **Neo4j 5.x** - Graph database (User relationships, Social graph)
- **Redis 7.0** - Cache & Session store

### AI/ML
- **PhoBERT** - Vietnamese BERT for text embeddings
  - Model: `vinai/phobert-base`
  - Embedding dimension: 768
  - Use cases: Content similarity, user profile matching

### DevOps
- **Docker & Docker Compose** - Containerization
- **Maven** - Java build tool
- **Git** - Version control
- **GitHub Actions** - CI/CD (optional)

---

## 💻 Yêu cầu hệ thống

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 20.04+)
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 20GB free space
- **Java**: JDK 17+
- **Python**: 3.11+
- **Node.js**: 18+
- **Docker**: 20.10+ (nếu dùng Docker)

### Recommended
- **CPU**: 8 cores
- **RAM**: 16GB
- **SSD**: 50GB
- **GPU**: NVIDIA GPU (cho PyTorch, optional)

---

## 🚀 Cài đặt và chạy

### Option 1: Docker Compose (Khuyến nghị)

#### Bước 1: Clone repository
```bash
git clone https://github.com/your-username/CTU-Connect-demo.git
cd CTU-Connect-demo
```

#### Bước 2: Cấu hình environment variables
```bash
# Copy file mẫu
cp .env.example .env

# Chỉnh sửa các biến môi trường
nano .env
```

#### Bước 3: Khởi động toàn bộ hệ thống
```bash
docker-compose up -d
```

Các service sẽ khởi động trên các port:
- Frontend: http://localhost:5173
- API Gateway: http://localhost:8090
- Eureka Dashboard: http://localhost:8761
- Neo4j Browser: http://localhost:7474
- MongoDB Express: http://localhost:8081 (optional)

#### Bước 4: Khởi tạo dữ liệu (Optional)
```bash
# Chạy migration script
docker-compose exec recommend-java mvn flyway:migrate

# Import sample data
python recommendation-service/data_migration.py
```

### Option 2: Local Development (Manual)

#### Bước 1: Cài đặt dependencies

**Java Services:**
```bash
# Cài đặt Maven dependencies cho tất cả services
cd auth-service && mvn clean install && cd ..
cd user-service && mvn clean install && cd ..
cd post-service && mvn clean install && cd ..
cd chat-service && mvn clean install && cd ..
cd media-service && mvn clean install && cd ..
cd api-gateway && mvn clean install && cd ..
cd eureka-server && mvn clean install && cd ..
cd recommend-service/java-api && mvn clean install && cd ../..
```

**Python Service:**
```bash
cd recommend-service/python-model
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd client
npm install
cd ../admin
npm install
```

#### Bước 2: Khởi động databases

**PostgreSQL:**
```bash
# Tạo databases
createdb auth_db
createdb media_db
createdb recommend_db
```

**MongoDB:**
```bash
# Khởi động MongoDB
mongod --dbpath /path/to/data

# Tạo databases
mongo
use post_db
use chat_db
```

**Neo4j:**
```bash
# Khởi động Neo4j
neo4j start
# Truy cập: http://localhost:7474
# Default credentials: neo4j/neo4j
```

**Redis:**
```bash
redis-server
```

**Kafka:**
```bash
# Khởi động Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Khởi động Kafka
bin/kafka-server-start.sh config/server.properties
```

#### Bước 3: Chạy các services

**Infrastructure Services:**
```bash
# Eureka Server
cd eureka-server
mvn spring-boot:run

# API Gateway
cd api-gateway
mvn spring-boot:run
```

**Core Services:**
```bash
# Auth Service
cd auth-service
mvn spring-boot:run

# User Service
cd user-service
mvn spring-boot:run

# Post Service
cd post-service
mvn spring-boot:run

# Chat Service
cd chat-service
mvn spring-boot:run

# Media Service
cd media-service
mvn spring-boot:run
```

**Recommendation Service:**
```bash
# Python AI Engine
cd recommend-service/python-model
source venv/bin/activate
uvicorn server:app --reload --port 8000

# Java Orchestrator
cd recommend-service/java-api
mvn spring-boot:run
```

**Frontend:**
```bash
# Client
cd client
npm run dev

# Admin
cd admin
npm run dev
```

---

## 📁 Cấu trúc thư mục

```
CTU-Connect-demo/
├── api-gateway/                 # API Gateway service
│   ├── src/main/java/
│   └── pom.xml
├── eureka-server/               # Service Discovery
│   ├── src/main/java/
│   └── pom.xml
├── auth-service/                # Authentication service
│   ├── src/main/java/
│   └── pom.xml
├── user-service/                # User management service
│   ├── src/main/java/
│   └── pom.xml
├── post-service/                # Post management service
│   ├── src/main/java/
│   └── pom.xml
├── chat-service/                # Real-time chat service
│   ├── src/main/java/
│   └── pom.xml
├── media-service/               # Media upload service
│   ├── src/main/java/
│   └── pom.xml
├── recommend-service/           # Recommendation service (Hybrid)
│   ├── java-api/               # Java Orchestrator
│   │   ├── src/main/java/
│   │   └── pom.xml
│   ├── python-model/           # Python AI Engine
│   │   ├── server.py          # FastAPI server
│   │   ├── inference.py       # PhoBERT inference
│   │   ├── model/             # PhoBERT model files
│   │   └── requirements.txt
│   └── docker/                # Docker configs
├── client/                     # React frontend (Client)
│   ├── src/
│   ├── public/
│   └── package.json
├── admin/                      # React frontend (Admin)
│   ├── src/
│   └── package.json
├── docker-compose.yml          # Docker Compose config
├── .env.example               # Environment variables template
└── README.md                  # This file
```

---

## 📚 API Documentation

### Base URLs
- **API Gateway**: `http://localhost:8090`
- **Auth Service**: `http://localhost:8091`
- **User Service**: `http://localhost:8092`
- **Post Service**: `http://localhost:8093`
- **Chat Service**: `http://localhost:8094`
- **Recommend Service**: `http://localhost:8095`
- **Media Service**: `http://localhost:8096`

### Authentication

Tất cả các API (trừ login/register) yêu cầu JWT token trong header:
```
Authorization: Bearer <token>
```

### Endpoints chính

#### 1. Auth Service (`/api/auth`)

**Register:**
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@student.ctu.edu.vn",
  "username": "testuser",
  "password": "Pass123!",
  "confirmPassword": "Pass123!"
}
```

**Login:**
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@student.ctu.edu.vn",
  "password": "Pass123!"
}
```

**Response:**
```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "tokenType": "Bearer",
  "expiresIn": 3600
}
```

#### 2. User Service (`/api/users`)

**Get Profile:**
```http
GET /api/users/profile
Authorization: Bearer <token>
```

**Update Profile:**
```http
PUT /api/users/me/profile
Authorization: Bearer <token>
Content-Type: application/json

{
  "fullName": "Nguyễn Văn A",
  "studentId": "B2110069",
  "major": "Công nghệ thông tin",
  "batch": "2021",
  "bio": "Yêu thích AI và Machine Learning"
}
```

**Friend Suggestions:**
```http
GET /api/users/friend-suggestions?limit=20
Authorization: Bearer <token>
```

**Send Friend Request:**
```http
POST /api/users/{userId}/friend-request
Authorization: Bearer <token>
```

#### 3. Post Service (`/api/posts`)

**Create Post:**
```http
POST /api/posts
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "Chia sẻ tài liệu Mạng máy tính",
  "title": "TCP/IP Protocol",
  "mediaIds": ["media-id-1", "media-id-2"],
  "visibility": "PUBLIC",
  "hashtags": ["MangMayTinh", "TCP", "CTU"]
}
```

**Get User Posts:**
```http
GET /api/posts/user/{userId}?page=0&size=20
Authorization: Bearer <token>
```

**Like Post:**
```http
POST /api/posts/{postId}/like
Authorization: Bearer <token>
```

**Comment:**
```http
POST /api/posts/{postId}/comments
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "Tài liệu rất hữu ích!"
}
```

#### 4. Recommendation Service (`/api/recommendations`)

**Get Personalized Feed:**
```http
GET /api/recommendations/feed?userId={userId}&size=20
Authorization: Bearer <token>
```

**Response:**
```json
{
  "recommendations": [
    {
      "postId": "post-123",
      "authorId": "user-456",
      "content": "Mạng máy tính chương 4...",
      "score": 0.8542,
      "contentSimilarity": 0.72,
      "academicScore": 0.90,
      "popularityScore": 0.65,
      "createdAt": "2025-12-10T10:30:00Z"
    }
  ],
  "count": 20,
  "userId": "user-123",
  "timestamp": "2025-12-12T15:45:00Z"
}
```

**Get Friend Suggestions:**
```http
GET /api/recommendations/friends?userId={userId}&limit=20
Authorization: Bearer <token>
```

**Response:**
```json
{
  "suggestions": [
    {
      "userId": "user-789",
      "fullName": "Trần Văn B",
      "mutualFriendsCount": 5,
      "suggestionReason": "5 bạn chung • Cùng ngành CNTT",
      "relevanceScore": 0.87,
      "suggestionType": "MUTUAL_FRIENDS"
    }
  ],
  "count": 20
}
```

#### 5. Chat Service (`/api/chat`)

**Get Conversations:**
```http
GET /api/chat/conversations
Authorization: Bearer <token>
```

**Get Messages:**
```http
GET /api/chat/conversations/{conversationId}/messages?page=0&size=50
Authorization: Bearer <token>
```

**WebSocket Connection:**
```javascript
const socket = io('ws://localhost:8094', {
  auth: {
    token: 'Bearer <jwt-token>'
  }
});

// Send message
socket.emit('message', {
  conversationId: 'conv-123',
  content: 'Hello!',
  type: 'TEXT'
});

// Receive message
socket.on('new_message', (message) => {
  console.log('New message:', message);
});
```

#### 6. Media Service (`/api/media`)

**Upload File:**
```http
POST /api/media/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <binary-data>
type: IMAGE
```

**Response:**
```json
{
  "mediaId": "media-123",
  "url": "http://localhost:8096/media/images/2025/12/image.jpg",
  "type": "IMAGE",
  "size": 2048576,
  "filename": "image.jpg"
}
```

### Postman Collection

Import Postman collection: [CTU-Connect.postman_collection.json](./docs/postman/CTU-Connect.postman_collection.json)

---

## 💾 Database Schema

### PostgreSQL (Recommend DB)

**Table: post_embeddings**
```sql
CREATE TABLE post_embeddings (
    id VARCHAR(36) PRIMARY KEY,
    post_id VARCHAR(36) UNIQUE NOT NULL,
    author_id VARCHAR(36) NOT NULL,
    content TEXT NOT NULL,
    title VARCHAR(500),
    embedding REAL[768] NOT NULL,
    author_major VARCHAR(100),
    author_faculty VARCHAR(100),
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Table: user_embeddings**
```sql
CREATE TABLE user_embeddings (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) UNIQUE NOT NULL,
    major VARCHAR(100),
    faculty VARCHAR(100),
    bio TEXT,
    interests TEXT[],
    embedding REAL[768] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Table: user_feedback**
```sql
CREATE TABLE user_feedback (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    post_id VARCHAR(36) NOT NULL,
    feedback_type VARCHAR(20) NOT NULL,
    feedback_value REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Neo4j (Graph DB)

**Nodes:**
- `User`: User profiles
- `Major`: Academic majors
- `Faculty`: Academic faculties
- `Batch`: Student batches

**Relationships:**
- `(User)-[:IS_FRIENDS_WITH]->(User)`
- `(User)-[:SENT_FRIEND_REQUEST_TO]->(User)`
- `(User)-[:ENROLLED_IN]->(Major)`
- `(User)-[:IN_BATCH]->(Batch)`
- `(Major)-[:HAS_MAJOR]-(Faculty)`

### MongoDB (Post DB)

**Collection: posts**
```javascript
{
  _id: ObjectId,
  authorId: String,
  content: String,
  title: String,
  mediaIds: [String],
  visibility: String, // PUBLIC, FRIENDS, PRIVATE
  hashtags: [String],
  likesCount: Number,
  commentsCount: Number,
  sharesCount: Number,
  createdAt: Date,
  updatedAt: Date
}
```

**Collection: comments**
```javascript
{
  _id: ObjectId,
  postId: String,
  authorId: String,
  content: String,
  parentCommentId: String, // for nested comments
  createdAt: Date
}
```

### MongoDB (Chat DB)

**Collection: conversations**
```javascript
{
  _id: ObjectId,
  participants: [String], // userIds
  lastMessage: String,
  lastMessageAt: Date,
  unreadCount: Object, // { userId: count }
  createdAt: Date
}
```

**Collection: messages**
```javascript
{
  _id: ObjectId,
  conversationId: String,
  senderId: String,
  content: String,
  type: String, // TEXT, IMAGE, FILE
  isRead: Boolean,
  readAt: Date,
  createdAt: Date
}
```

---

## 🧪 Testing

### Unit Tests

**Java Services:**
```bash
# Run all tests
mvn test

# Run specific service tests
cd auth-service && mvn test
cd user-service && mvn test
```

**Python Service:**
```bash
cd recommend-service/python-model
pytest tests/
```

### Integration Tests

```bash
# Run integration tests
mvn verify -P integration-tests
```

### API Tests (Postman/Newman)

```bash
# Install Newman
npm install -g newman

# Run Postman collection
newman run docs/postman/CTU-Connect.postman_collection.json \
  --environment docs/postman/local.postman_environment.json
```

### Load Testing (K6)

```bash
# Install k6
brew install k6  # macOS
choco install k6  # Windows

# Run load test
k6 run tests/load/feed-test.js
```

### Test Coverage

| Service | Coverage | Status |
|---------|----------|--------|
| Auth Service | 85% | ✅ |
| User Service | 82% | ✅ |
| Post Service | 78% | ✅ |
| Chat Service | 75% | ⚠️ |
| Media Service | 80% | ✅ |
| Recommend Service (Java) | 70% | ⚠️ |
| Recommend Service (Python) | 65% | ⚠️ |

---

## 🚢 Deployment

### Docker Deployment

#### Production Build
```bash
# Build all services
docker-compose -f docker-compose.prod.yml build

# Push to registry
docker-compose -f docker-compose.prod.yml push
```

#### Deploy to Server
```bash
# SSH to server
ssh user@your-server.com

# Pull and run
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/secrets/
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n ctu-connect
kubectl get services -n ctu-connect
```

### Environment Variables

**Production `.env`:**
```bash
# Database
POSTGRES_HOST=prod-db.example.com
MONGODB_URI=mongodb://prod-mongo.example.com:27017
NEO4J_URI=bolt://prod-neo4j.example.com:7687
REDIS_HOST=prod-redis.example.com

# JWT
JWT_SECRET=your-super-secret-key-change-this-in-production
JWT_EXPIRATION=3600000

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# AWS S3 (if using)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_S3_BUCKET=ctu-connect-media

# Python Service
PYTHON_SERVICE_URL=http://recommend-python:8000

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka1:9092,kafka2:9092,kafka3:9092
```

---

## 📖 Documentation

- [Architecture Documentation](./docs/ARCHITECTURE.md)
- [API Documentation](./docs/API.md)
- [Database Schema](./docs/DATABASE.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)
- [Contributing Guide](./CONTRIBUTING.md)

---

## 🤝 Contributors

### Development Team

| Name | Role | GitHub | Email |
|------|------|--------|-------|
| **Nguyễn Văn A** | Full-stack Developer | [@nguyenvana](https://github.com/nguyenvana) | baoB2110069@student.ctu.edu.vn |
| **Trần Thị B** | Backend Developer | [@tranthib](https://github.com/tranthib) | tranb@student.ctu.edu.vn |
| **Lê Văn C** | Frontend Developer | [@levanc](https://github.com/levanc) | lec@student.ctu.edu.vn |

### Advisors

- **TS. Nguyễn Xuân Huy** - Project Supervisor
- **ThS. Võ Thị Kim Anh** - Technical Advisor

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Đại học Cần Thơ** - Hỗ trợ tài nguyên và cơ sở vật chất
- **VinAI Research** - PhoBERT model
- **HuggingFace** - Transformers library
- **Spring Team** - Spring Boot framework
- **FastAPI Team** - FastAPI framework
- **Neo4j Team** - Graph database platform

---

## 📞 Contact & Support

- **Website**: [https://ctu-connect.example.com](https://ctu-connect.example.com)
- **Email**: support@ctu-connect.example.com
- **GitHub Issues**: [https://github.com/your-username/CTU-Connect-demo/issues](https://github.com/your-username/CTU-Connect-demo/issues)
- **Slack Community**: [Join our Slack](https://ctu-connect.slack.com)

---

## 🎓 Academic Publication

Nếu bạn sử dụng CTU-Connect trong nghiên cứu, vui lòng cite:

```bibtex
@misc{ctuconnect2025,
  title={CTU-Connect: An AI-Powered Academic Social Network for Can Tho University},
  author={Nguyen, Van A and Tran, Thi B and Le, Van C},
  year={2025},
  institution={Can Tho University},
  howpublished={\url{https://github.com/your-username/CTU-Connect-demo}}
}
```

---

<div align="center">
  <p>Made with ❤️ by CTU-Connect Team</p>
  <p>© 2025 Can Tho University. All rights reserved.</p>
</div>
