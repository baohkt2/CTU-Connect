# ⚠️ QUAN TRỌNG: CHẠY HYBRID ARCHITECTURE

**Ngày:** 07/12/2025 18:24

---

## ❓ CÂU HỎI CỦA BẠN

> "Vấn đề là hiện tại tôi nên chạy cái nào? 
> Chạy recommendation-service hay recommendation-service-java + recommendation-service-python?"

---

## ✅ TRẢ LỜI NGẮN GỌN

**CHẠY: recommendation-service-java + recommendation-service-python**

**KHÔNG CHẠY: recommendation-service (legacy, duplicate)**

---

## 🔍 GIẢI THÍCH

### Tình trạng hiện tại:

Hiện có **3 thư mục** recommendation:

1. **recommendation-service** (OLD/DUPLICATE)
   - ❌ Thư mục cũ hoặc duplicate
   - ❌ Code giống hệt recommendation-service-java
   - ❌ KHÔNG NÊN SỬ DỤNG

2. **recommendation-service-java** (CHÍNH THỨC - JAVA)
   - ✅ Java Spring Boot service
   - ✅ API endpoints
   - ✅ Business logic
   - ✅ Database integration
   - ✅ Gọi Python service qua HTTP
   - ✅ Port: 8095

3. **recommendation-service-python** (CHÍNH THỨC - PYTHON)
   - ✅ Python FastAPI service
   - ✅ Machine Learning models
   - ✅ PhoBERT embeddings
   - ✅ ML ranking
   - ✅ Port: 8097

---

## 🎯 KIẾN TRÚC ĐÚNG

```
                     HYBRID ARCHITECTURE
                            
┌─────────────────────────────────────────────────┐
│                                                 │
│  recommendation-service-java (Port 8095)        │
│  ────────────────────────────────────────       │
│  • Java Spring Boot                             │
│  • REST API Endpoints                           │
│  • Business Logic                               │
│  • Database (PostgreSQL, Neo4j, Redis)          │
│  • Kafka Integration                            │
│  • Caching                                      │
│  • ⭐ Gọi Python service qua HTTP               │
│                                                 │
└────────────────┬────────────────────────────────┘
                 │
                 │ HTTP Request
                 │ POST /api/model/predict
                 ↓
┌─────────────────────────────────────────────────┐
│                                                 │
│  recommendation-service-python (Port 8097)      │
│  ──────────────────────────────────────────     │
│  • Python FastAPI                               │
│  • Machine Learning Models                      │
│  • PhoBERT (Vietnamese NLP)                     │
│  • Text Embeddings                              │
│  • Similarity Calculation                       │
│  • ML-based Ranking                             │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 📋 BƯỚC CHẠY ĐÚNG

### Bước 1: Start Databases

```powershell
cd D:\LVTN\CTU-Connect-demo\recommendation-service-java
docker-compose -f docker-compose.dev.yml up -d
```

**Verify:**
```powershell
docker ps
# Phải thấy: postgres, neo4j, redis, kafka
```

---

### Bước 2: Start Java Service

**Option A: Chạy trong IntelliJ IDEA** (Khuyên dùng)

1. Open project: \ecommendation-service-java\
2. Wait for Maven sync
3. Run: \RecommendationServiceApplication\
4. Profile: \dev\

**Option B: Chạy bằng Maven**

```powershell
cd D:\LVTN\CTU-Connect-demo\recommendation-service-java
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

**Verify:**
```powershell
curl http://localhost:8095/actuator/health
# Expected: {"status":"UP"}
```

---

### Bước 3: Start Python Service

```powershell
cd D:\LVTN\CTU-Connect-demo\recommendation-service-python

# Tạo virtual environment (chỉ lần đầu)
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies (chỉ lần đầu hoặc khi có thay đổi)
pip install -r requirements.txt

# Run service
python app.py
```

**Output mong đợi:**
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8097
INFO:     Application startup complete.
```

**Verify:**
```powershell
curl http://localhost:8097/health
# Expected: {"status":"healthy"}

# API docs
Start-Process http://localhost:8097/docs
```

---

### Bước 4: Test Integration

```powershell
# Test Java API (sẽ gọi Python bên trong)
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=10"
```

**Logs Java sẽ hiện:**
```
INFO - Getting feed for user: user123
INFO - Calling Python model service...
INFO - Python model returned 10 ranked posts
INFO - Processing time: 245ms
```

**Logs Python sẽ hiện:**
```
INFO - POST /api/model/predict
INFO - Processing recommendation for user: user123
INFO - Ranked 10 posts
INFO - Response time: 230ms
```

---

## 🚫 KHÔNG LÀM GÌ VỚI recommendation-service

### Tại sao có thư mục này?

Có thể là:
1. **Backup cũ** trước khi refactor
2. **Duplicate** do copy nhầm
3. **Legacy code** chưa xóa

### Kiểm chứng:

```powershell
# Code giống HỆT nhau
PS> $(Get-FileHash recommendation-service\src\main\java\vn\ctu\edu\recommend\service\HybridRecommendationService.java).Hash
PS> $(Get-FileHash recommendation-service-java\src\main\java\vn\ctu\edu\recommend\service\HybridRecommendationService.java).Hash
# Kết quả: GIỐNG NHAU (MD5 hash identical)
```

### Nên làm gì?

**Option 1: Xóa thư mục (Khuyên dùng)**
```powershell
# Backup trước
Rename-Item recommendation-service recommendation-service-backup

# Hoặc xóa hẳn (sau khi chắc chắn)
Remove-Item -Recurse -Force recommendation-service
```

**Option 2: Giữ lại nhưng đổi tên**
```powershell
Rename-Item recommendation-service recommendation-service-old-do-not-use
```

---

## 📊 DOCKER-COMPOSE HIỆN TẠI

File \docker-compose.yml\ đã config đúng:

```yaml
services:
  recommendation-service:
    build: ./recommendation-service-java  # ← ĐÚNG RỒI!
    ports:
      - "8095:8095"
    environment:
      - PYTHON_MODEL_SERVICE_URL=http://python-model-service:8097
```

**Chưa có:** Python service trong docker-compose.yml (cần thêm)

---

## ✅ CHECKLIST

Để chạy recommendation system đúng:

- [ ] Start databases (docker-compose)
- [ ] Start Java service từ \ecommendation-service-java/\
- [ ] Start Python service từ \ecommendation-service-python/\
- [ ] Test Java API: http://localhost:8095/actuator/health
- [ ] Test Python API: http://localhost:8097/health
- [ ] Test integration: Get recommendations qua Java

**KHÔNG:**
- [x] ~~Chạy recommendation-service~~ ← CŨ, BỎ QUA
- [x] ~~Chạy chỉ Java không có Python~~ ← Thiếu ML
- [x] ~~Chạy chỉ Python không có Java~~ ← Thiếu API

---

## 🎯 TÓM TẮT

### CẦN CHẠY (HYBRID):

1. **Java Service** ← recommendation-service-java/
   - Port 8095
   - API + Business Logic + DB

2. **Python Service** ← recommendation-service-python/
   - Port 8097
   - ML + PhoBERT + Ranking

### KHÔNG CHẠY:

3. ~~**recommendation-service**~~ ← Legacy/Duplicate

---

## 📝 NEXT STEPS

1. ✅ Xóa hoặc rename \ecommendation-service\ để tránh nhầm lẫn
2. ✅ Thêm Python service vào \docker-compose.yml\
3. ✅ Chạy test script: \	est-hybrid-recommendation.ps1\
4. ✅ Load test data
5. ✅ Train ML models (optional, có fallback)

---

## 📚 TÀI LIỆU THAM KHẢO

- **Setup Guide:** RECOMMENDATION_HYBRID_SETUP.md
- **Architecture:** recommendation-service-java/HYBRID_ARCHITECTURE.md
- **Index:** RECOMMENDATION_INDEX.md
- **Explanation:** RECOMMENDATION_ARCHITECTURE_EXPLAINED_VN.md

---

**✅ KẾT LUẬN:**

**CHẠY 2 SERVICES:**
- recommendation-service-java (Java) Port 8095
- recommendation-service-python (Python) Port 8097

**BỎ QUA:**
- recommendation-service (duplicate/legacy)

---

**Tạo bởi:** GitHub Copilot CLI  
**Ngày:** 07/12/2025 18:24

