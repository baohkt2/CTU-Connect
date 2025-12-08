# 🚀 START HERE - Recommendation Service

## 👋 Chào mừng!

Bạn đang ở hệ thống **Recommendation Service** của CTU Connect - một hệ thống gợi ý nội dung học thuật thông minh sử dụng PhoBERT và kiến trúc hybrid Python-Java.

---

## 🎯 Bạn muốn làm gì?

### 1️⃣ Tôi muốn khởi động hệ thống ngay (5 phút)

👉 **[QUICKSTART.md](./QUICKSTART.md)**

```bash
cd recommend-service/docker
docker-compose up -d
```

---

### 2️⃣ Tôi muốn hiểu kiến trúc hệ thống

👉 **[ARCHITECTURE.md](./ARCHITECTURE.md)**

Tìm hiểu về:
- Kiến trúc 3 layers- PhoBERT model integration
- Luồng hoạt động chi tiết
- Python + Java hybrid architecture

---

### 3️⃣ Tôi muốn tổng quan toàn bộ dự án

👉 **[README.md](./README.md)**

Bao gồm:
- Tính năng chính
- Hướng dẫn cài đặt
- API documentation
- Configuration guide
- Testing & troubleshooting

---

### 4️⃣ Tôi muốn navigate tất cả tài liệu

👉 **[INDEX.md](./INDEX.md)**

Navigation đầy đủ:
- Quick links
- Learning path
- API reference
- Resources

---

### 5️⃣ Tôi cần migrate từ old services

👉 **[MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)**

Hướng dẫn chi tiết:
- File mapping old → new
- Step-by-step migration
- Code changes
- Testing & rollback

---

### 6️⃣ Tôi muốn biết có gì thay đổi

👉 **[RESTRUCTURE_SUMMARY.md](./RESTRUCTURE_SUMMARY.md)**

Chi tiết:
- Before/after comparison
- Benefits achieved
- Developer notes
- Verification checklist

---

## 🗂 Cấu trúc dự án

```
recommend-service/
│
├── 📖 START_HERE.md              ← BẠN ĐANG Ở ĐÂY
│
├── 📚 Documentation
│   ├── QUICKSTART.md             ⭐ Bắt đầu nhanh
│   ├── README.md                 ⭐ Tổng quan
│   ├── ARCHITECTURE.md           ⭐ Kiến trúc
│   ├── INDEX.md                  📑 Navigation
│   ├── MIGRATION_GUIDE.md        🔄 Migration
│   └── RESTRUCTURE_SUMMARY.md    📋 Summary
│
├── 🐍 python-model/              Python AI Service
│   ├── inference.py              ⚡ Core AI Engine
│   ├── server.py                 🌐 FastAPI Server
│   ├── model/                    🧠 PhoBERT Model
│   └── requirements.txt
│
├── ☕ java-api/                  Java API Service
│   ├── src/                      💼 Source Code
│   └── pom.xml
│
└── 🐳 docker/                    Docker Configs
    ├── docker-compose.yml        🎭 Orchestration
    ├── recommend-java.Dockerfile
    └── recommend-python.Dockerfile
```

---

## 🎓 Learning Path

### Beginner (15 phút)
1. Đọc file này (START_HERE.md)
2. Xem [QUICKSTART.md](./QUICKSTART.md)
3. Khởi động hệ thống với Docker
4. Test một vài API endpoints

### Intermediate (1 giờ)
1. Đọc [README.md](./README.md)
2. Tìm hiểu API documentation
3. Test với Postman/curl
4. Xem logs và monitoring

### Advanced (2-3 giờ)
1. Đọc [ARCHITECTURE.md](./ARCHITECTURE.md)
2. Hiểu luồng hoạt động chi tiết
3. Xem source code
4. Chạy tests
5. Customize configuration

---

## 🚀 Quick Commands

### Docker (Khuyến nghị)

```bash
# Start services
cd docker && docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Development

```bash
# Python service
cd python-model
uvicorn server:app --reload --port 8000

# Java service
cd java-api
./mvnw spring-boot:run
```

### Testing

```bash
# Health checks
curl http://localhost:8000/health        # Python
curl http://localhost:8081/actuator/health  # Java

# Test recommendation
curl "http://localhost:8081/api/recommendations/feed?userId=test"
```

---

## 🌟 Key Features

### AI-Powered Recommendations
- ✅ PhoBERT semantic understanding
- ✅ Vietnamese text processing
- ✅ Personalized content
- ✅ Real-time updates

### Academic Focus
- ✅ Major/faculty matching
- ✅ Course recommendations
- ✅ Study materials
- ✅ Student connections

### High Performance
- ✅ Redis caching
- ✅ Batch processing
- ✅ Async operations
- ✅ Horizontal scaling

---

## 📊 API Overview

### Python Service (Port 8000)
```
POST /embed/post              - Generate post embedding
POST /embed/post/batch        - Batch embeddings
POST /embed/user              - User profile embedding
POST /similarity              - Compute similarity
GET  /health                  - Health check
```

### Java Service (Port 8081)
```
GET  /api/recommendations/feed       - Personalized feed
GET  /api/recommendations/academic   - Academic posts
GET  /api/recommendations/users      - User suggestions
POST /api/recommendations/refresh    - Refresh embeddings
```

---

## ❓ Common Questions

### Q: Tôi nên bắt đầu từ đâu?
**A:** Xem [QUICKSTART.md](./QUICKSTART.md) để khởi động trong 5 phút.

### Q: Làm sao để hiểu kiến trúc?
**A:** Đọc [ARCHITECTURE.md](./ARCHITECTURE.md) - có diagrams và giải thích chi tiết.

### Q: API documentation ở đâu?
**A:** Xem phần API trong [README.md](./README.md) hoặc truy cập Swagger UI tại `http://localhost:8000/docs`.

### Q: Làm sao để test?
**A:** Xem Testing section trong [README.md](./README.md#testing).

### Q: Gặp lỗi thì làm gì?
**A:** Xem Troubleshooting trong [QUICKSTART.md](./QUICKSTART.md#troubleshooting).

### Q: Có video hướng dẫn không?
**A:** Chưa có, nhưng documentation rất chi tiết và có examples.

---

## 🔗 External Links

### Technologies Used
- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) - Vietnamese BERT
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [Spring Boot](https://spring.io/projects/spring-boot) - Java framework
- [Docker](https://www.docker.com/) - Containerization

### Related Services
- Post Service - Manages posts
- User Service - User management
- API Gateway - Routing
- Eureka Server - Service discovery

---

## 🆘 Getting Help

### Documentation
- ✅ All docs in this directory
- ✅ Examples included
- ✅ Troubleshooting guides

### Support Channels
- 📧 Email: dev@ctuconnect.edu.vn
- 💬 Slack: #recommendation-service
- 🐛 GitHub Issues

### Resources
- Team Wiki
- API Postman Collection
- Architecture Diagrams
- Code Examples

---

## ✅ Checklist cho người mới

Trước khi bắt đầu development:

- [ ] Đã đọc START_HERE.md (file này)
- [ ] Đã xem QUICKSTART.md
- [ ] Đã khởi động được services
- [ ] Đã test health endpoints
- [ ] Đã đọc ARCHITECTURE.md
- [ ] Đã hiểu API endpoints
- [ ] Đã test một vài requests
- [ ] Đã xem logs
- [ ] Đã join Slack channel
- [ ] Sẵn sàng code! 🚀

---

## 🎯 Next Steps

1. **Khởi động ngay** → [QUICKSTART.md](./QUICKSTART.md)
2. **Tìm hiểu kiến trúc** → [ARCHITECTURE.md](./ARCHITECTURE.md)
3. **Xem tổng quan** → [README.md](./README.md)
4. **Browse docs** → [INDEX.md](./INDEX.md)

---

## 📝 Notes

- ⚡ Quick start chỉ mất **5 phút**
- 📚 Documentation đầy đủ **60KB+**
- 🐳 Docker setup **1 command**
- 🎓 Learning curve **vừa phải**
- 🚀 Production ready **✅**

---

## 🎉 Welcome to the Team!

Chúc bạn có trải nghiệm tốt với Recommendation Service!

Nếu có câu hỏi, đừng ngại hỏi team hoặc tham khảo documentation.

**Happy Coding! 💻**

---

*Last updated: December 2024*  
*Version: 1.0.0*  
*Status: ✅ Complete & Ready*
