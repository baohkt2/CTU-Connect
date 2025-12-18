# 🎓 CTU-Connect

> **Mạng xã hội học thuật thông minh dành cho sinh viên và giảng viên Đại học Cần Thơ**

[![Java](https://img.shields.io/badge/Java-17-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white)](https://www.java.com/)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.2-6DB33F?style=for-the-badge&logo=spring-boot)](https://spring.io/projects/spring-boot)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.13-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng chính](#-tính-năng-chính)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt và chạy](#-cài-đặt-và-chạy)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [API Documentation](#-api-documentation)
- [Đóng góp](#-đóng-góp)
- [License](#-license)

---

## 🎯 Giới thiệu

**CTU-Connect** là nền tảng mạng xã hội học thuật được xây dựng dành riêng cho cộng đồng Đại học Cần Thơ. Hệ thống tích hợp công nghệ **AI/Machine Learning** sử dụng **PhoBERT** (Vietnamese BERT) để cung cấp trải nghiệm cá nhân hóa thông minh.

### 🎯 Vấn đề giải quyết

| Vấn đề | Giải pháp CTU-Connect |
|--------|----------------------|
| 📚 Khó khăn chia sẻ tài liệu học tập | Nền tảng chia sẻ tài liệu theo chuyên ngành |
| 🤝 Thiếu kết nối học thuật | Gợi ý kết bạn thông minh dựa trên ML |
| 📰 Quá tải thông tin | News Feed cá nhân hóa với AI |
| 💬 Giao tiếp phân tán | Chat real-time tích hợp |

---

## ✨ Tính năng chính

### 📝 Quản lý bài viết
- Đăng bài với text, hình ảnh, hashtags
- Like, Comment, Share
- Visibility control (Public, Friends, Private)

### 📰 News Feed thông minh (AI-Powered)
- **Hybrid Recommendation Engine**:
  - Content-based filtering với PhoBERT embeddings (35%)
  - Implicit feedback từ lịch sử tương tác (25%)
  - Academic relevance (cùng ngành, khoa) (25%)
  - Popularity score (15%)
- Cache thông minh với Redis

### 👥 Gợi ý kết bạn (ML-Enhanced)
- PhoBERT similarity trên profile
- Mutual friends analysis
- Academic connections
- Lý do gợi ý rõ ràng

### 💬 Chat Real-time
- WebSocket messaging
- Typing indicators
- Online/Offline presence
- Message persistence

### 🔐 Xác thực & Bảo mật
- JWT-based authentication
- Email verification
- reCAPTCHA v3 protection
- Role-based access control

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                                  │
│         Next.js 15 (Client :3000 | Admin :3001)                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (:8090)                           │
│              Spring Cloud Gateway + Service Discovery            │
└─────┬────────────┬──────────┬──────────┬──────────┬────────────┘
      │            │          │          │          │
      ▼            ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│   Auth   │ │   User   │ │   Post   │ │   Chat   │ │  Media   │
│ Service  │ │ Service  │ │ Service  │ │ Service  │ │ Service  │
│  :8080   │ │  :8081   │ │  :8085   │ │  :8086   │ │  :8084   │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │            │
     ▼            ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│PostgreSQL│ │  Neo4j   │ │ MongoDB  │ │ MongoDB  │ │PostgreSQL│
│ auth_db  │ │ Graph DB │ │ post_db  │ │ chat_db  │ │ media_db │
│  :5433   │ │  :7687   │ │  :27018  │ │  :27019  │ │  :5434   │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘

┌─────────────────────────────────────────────────────────────────┐
│              RECOMMENDATION SERVICE (Hybrid Architecture)        │
│  ┌────────────────────────┐      ┌──────────────────────────┐  │
│  │   Java Orchestrator    │◄────►│   Python AI Engine       │  │
│  │   Port: 8095           │ HTTP │   Port: 8000             │  │
│  │  - Business Logic      │      │   - PhoBERT Model        │  │
│  │  - Cache Management    │      │   - Embedding Generation │  │
│  └───────┬────────────────┘      └──────────────────────────┘  │
│          │                                                      │
│          ▼                                                      │
│  ┌──────────────────┐        ┌──────────────────┐             │
│  │   PostgreSQL     │        │      Redis       │             │
│  │   recommend_db   │        │   Cache :6380    │             │
│  │      :5435       │        └──────────────────┘             │
│  └──────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Kafka    │  │    Redis    │  │   Eureka    │             │
│  │  Event Bus  │  │ Global Cache│  │  Discovery  │             │
│  │    :9092    │  │    :6379    │  │    :8761    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Công nghệ sử dụng

### Backend
| Công nghệ | Phiên bản | Mô tả |
|-----------|-----------|-------|
| Java | 17 | Main language |
| Spring Boot | 3.2.x | Core framework |
| Spring Cloud Gateway | - | API Gateway |
| Spring Cloud Netflix Eureka | - | Service Discovery |
| Spring Data JPA | - | ORM |
| Spring Data Neo4j | - | Graph database |
| Spring Kafka | - | Event streaming |
| Spring WebSocket | - | Real-time communication |
| FastAPI | - | Python AI service |
| PyTorch + Transformers | - | PhoBERT model |

### Frontend
| Công nghệ | Phiên bản | Mô tả |
|-----------|-----------|-------|
| Next.js | 15.x | React framework |
| React | 19.x | UI library |
| TailwindCSS | 4.x | Styling |
| TanStack Query | 5.x | Data fetching |
| STOMP.js | 7.x | WebSocket client |

### Databases
| Database | Phiên bản | Sử dụng cho |
|----------|-----------|-------------|
| PostgreSQL | 15 | Auth, Media, Recommend |
| MongoDB | 7.0 | Posts, Chat |
| Neo4j | 5.13 | User relationships |
| Redis | 7 | Caching, Sessions |

### Infrastructure
| Tool | Mô tả |
|------|-------|
| Docker & Docker Compose | Containerization |
| Apache Kafka | Event streaming |
| Cloudinary | Media storage |

---

## 💻 Yêu cầu hệ thống

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, Linux
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 20GB free space
- **Docker**: 20.10+ với Docker Compose

### Recommended
- **CPU**: 8+ cores
- **RAM**: 16GB
- **SSD**: 50GB

---

## 🚀 Cài đặt và chạy

### Bước 1: Clone repository

```bash
git clone https://github.com/your-username/CTU-Connect.git
cd CTU-Connect
```

### Bước 2: Cấu hình environment variables

```bash
# Copy file mẫu
cp .env.example .env

# Chỉnh sửa các biến môi trường (BẮT BUỘC)
# - Thay đổi tất cả password và secret key
# - Cấu hình MAIL_USERNAME, MAIL_PASSWORD (Gmail App Password)
# - Cấu hình CLOUDINARY credentials
# - Cấu hình RECAPTCHA keys
```

### Bước 3: Khởi động với Docker Compose

```bash
# Khởi động toàn bộ hệ thống
docker-compose up -d

# Theo dõi logs
docker-compose logs -f
```

### Bước 4: Truy cập ứng dụng

| Service | URL |
|---------|-----|
| Client Frontend | http://localhost:3000 |
| Admin Frontend | http://localhost:3001 |
| API Gateway | http://localhost:8090 |
| Eureka Dashboard | http://localhost:8761 |
| Neo4j Browser | http://localhost:7474 |

### Chạy Frontend (Development)

```bash
# Client Frontend
cd client-frontend
npm install
npm run dev

# Admin Frontend
cd admin-frontend
npm install
npm run dev
```

---

## 📁 Cấu trúc thư mục

```
CTU-Connect/
├── api-gateway/                 # API Gateway (Spring Cloud Gateway)
├── eureka-server/               # Service Discovery (Netflix Eureka)
├── auth-service/                # Authentication & Authorization
├── user-service/                # User Management (Neo4j)
├── post-service/                # Post Management (MongoDB)
├── chat-service/                # Real-time Chat (MongoDB + WebSocket)
├── media-service/               # Media Upload (Cloudinary)
├── recommend-service/           # AI Recommendation
│   ├── java-api/               # Java Orchestrator
│   └── python-model/           # Python AI Engine (PhoBERT)
├── client-frontend/             # Next.js Client App
├── admin-frontend/              # Next.js Admin App
├── database/                    # Database init scripts
│   ├── auth_db/                # PostgreSQL init
│   ├── media_db/               # PostgreSQL init
│   └── neo4j/                  # Neo4j init
├── docker-compose.yml           # Docker Compose configuration
├── .env.example                # Environment template
└── README.md
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8090/api
```

### Authentication
Tất cả API (trừ login/register) yêu cầu JWT token:
```
Authorization: Bearer <token>
```

### Main Endpoints

#### Auth Service
```http
POST /api/auth/register    # Đăng ký
POST /api/auth/login       # Đăng nhập
POST /api/auth/refresh     # Refresh token
POST /api/auth/verify      # Xác thực email
```

#### User Service
```http
GET  /api/users/profile         # Lấy profile
PUT  /api/users/me/profile      # Cập nhật profile
GET  /api/users/friend-suggestions  # Gợi ý kết bạn
POST /api/users/{id}/friend-request # Gửi lời mời kết bạn
GET  /api/users/friends         # Danh sách bạn bè
```

#### Post Service
```http
GET  /api/posts                 # Lấy posts
POST /api/posts                 # Tạo post
GET  /api/posts/{id}            # Chi tiết post
POST /api/posts/{id}/like       # Like post
POST /api/posts/{id}/comments   # Comment
```

#### Chat Service
```http
GET  /api/chat/conversations    # Danh sách conversations
GET  /api/chat/conversations/{id}/messages  # Lấy messages
WebSocket: /ws/chat             # Real-time messaging
```

#### Recommendation Service
```http
GET /api/recommendations/feed           # Personalized feed
GET /api/recommendations/friends        # Friend suggestions
```

---

## 🔐 Bảo mật

### Các biến môi trường cần bảo mật

⚠️ **QUAN TRỌNG**: Không commit file `.env` lên repository!

| Biến | Mô tả |
|------|-------|
| `JWT_SECRET` | Secret key cho JWT signing |
| `POSTGRES_PASSWORD` | Database password |
| `NEO4J_PASSWORD` | Neo4j password |
| `MAIL_PASSWORD` | Gmail App Password |
| `CLOUDINARY_API_SECRET` | Cloudinary secret |
| `RECAPTCHA_SECRET_KEY` | reCAPTCHA secret |

### Tạo JWT Secret Key
```bash
openssl rand -base64 32
```

### Gmail App Password
1. Bật 2-Factor Authentication trên Google Account
2. Truy cập: https://myaccount.google.com/apppasswords
3. Tạo App Password cho "Mail"

---

## 🤝 Đóng góp

Xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết hướng dẫn đóng góp.

### Quick Start
1. Fork repository
2. Tạo branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Tạo Pull Request

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## 👥 Tác giả

**Luận văn tốt nghiệp** - Đại học Cần Thơ

---

## 🙏 Acknowledgments

- [VinAI Research](https://github.com/VinAIResearch) - PhoBERT model
- [Spring Team](https://spring.io/) - Spring Boot framework
- [Neo4j](https://neo4j.com/) - Graph database
- [Vercel](https://vercel.com/) - Next.js framework

---

<div align="center">
  <p>Made with ❤️ for Can Tho University</p>
</div>
