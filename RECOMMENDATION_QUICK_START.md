# ⚡ RECOMMENDATION SERVICE - QUICK START GUIDE

## 🎯 Mục đích

Hướng dẫn nhanh để setup và chạy Recommendation Service trong môi trường development (trên IDE).

---

## 📋 Tóm tắt

Recommendation Service gồm **2 phần**:

1. **Java Service** (Port 8095) - API Gateway, Business Logic
2. **Python Service** (Port 8097) - Machine Learning Engine

**Luồng hoạt động:**
```
Client → Java Service → Python ML Service → Trả về recommendations
         ↓              ↓
    PostgreSQL      PhoBERT Model
    Neo4j           ML Ranking
    Redis
```

---

## 🚀 SETUP NHANH (5 bước)

### Bước 1: Start Databases (Docker)

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java
docker-compose -f docker-compose.dev.yml up -d
```

Kiểm tra:
```powershell
docker ps  # Phải thấy: postgres, neo4j, redis, kafka
```

---

### Bước 2: Setup Python Service

```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-python

# Tạo virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies (mất ~5-10 phút)
pip install -r requirements.txt
```

---

### Bước 3: Chạy Python Service

```powershell
# Đảm bảo venv đã activate
python app.py
```

Kiểm tra:
```powershell
curl http://localhost:8097/health
```

Phải trả về:
```json
{"status": "healthy", "service": "python-ml-service"}
```

---

### Bước 4: Chạy Java Service (IntelliJ)

1. Mở `recommendation-service-java` trong IntelliJ
2. Đợi Maven import xong
3. Tìm file `RecommendationServiceApplication.java`
4. Click Run ▶️

Hoặc chạy bằng Maven:
```powershell
cd d:\LVTN\CTU-Connect-demo\recommendation-service-java
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

Kiểm tra:
```powershell
curl http://localhost:8095/actuator/health
```

---

### Bước 5: Test toàn bộ hệ thống

```powershell
cd d:\LVTN\CTU-Connect-demo
.\test-recommendation-dev.ps1
```

---

## ✅ Checklist

- [ ] Docker Desktop đang chạy
- [ ] Các containers: postgres, neo4j, redis, kafka UP
- [ ] Python service (8097) chạy và trả về health OK
- [ ] Java service (8095) chạy và trả về health OK
- [ ] Test script chạy thành công (> 80% pass)

---

## 🔗 Liên kết quan trọng

| Service | URL | Description |
|---------|-----|-------------|
| Python API Docs | http://localhost:8097/docs | Swagger UI |
| Python Health | http://localhost:8097/health | Health check |
| Java Health | http://localhost:8095/actuator/health | Health check |
| Neo4j Browser | http://localhost:7474 | Graph database |

---

## 🐛 Troubleshooting nhanh

### Python không start?

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Reinstall
pip install -r requirements.txt

# Check Python version (cần 3.10+)
python --version
```

### Java không kết nối Python?

Check Python đang chạy:
```powershell
curl http://localhost:8097/health
```

Check config trong `application-dev.yml`:
```yaml
recommendation:
  python-service:
    url: http://localhost:8097
    enabled: true
```

### Database connection failed?

```powershell
# Restart containers
docker-compose -f docker-compose.dev.yml restart

# Check status
docker ps
```

### Port đã được sử dụng?

```powershell
# Check port
netstat -ano | findstr "8097"
netstat -ano | findstr "8095"

# Kill process
taskkill /PID <PID> /F
```

---

## 📚 Chi tiết hơn?

Xem file: **`RECOMMENDATION_DEV_SETUP_VN.md`** để có hướng dẫn chi tiết đầy đủ.

---

## 🎯 Test Endpoints

### Python Service

```powershell
# Health check
curl http://localhost:8097/health

# Prediction test
curl -X POST http://localhost:8097/api/model/predict `
  -H "Content-Type: application/json" `
  -d '{
    "userAcademic": {"userId": "user123"},
    "candidatePosts": [],
    "topK": 10
  }'
```

### Java Service

```powershell
# Health check
curl http://localhost:8095/actuator/health

# Get recommendations
curl "http://localhost:8095/api/recommendation/feed?userId=user123&size=10"

# Get similar posts
curl "http://localhost:8095/api/recommendation/similar/post123?size=5"
```

---

## 🔄 Workflow phát triển

1. **Sửa Python code** → Python tự reload (nếu DEBUG=True)
2. **Sửa Java code** → Restart từ IntelliJ
3. **Clear cache** → `docker exec redis redis-cli FLUSHDB`
4. **Test** → Chạy `.\test-recommendation-dev.ps1`

---

## 📊 Monitoring

### Logs

**Python:**
```powershell
# Xem trong terminal đang chạy Python
# Hoặc
Get-Content recommendation-service-python\logs\*.log -Tail 50
```

**Java:**
```powershell
# Xem trong IntelliJ Console
```

### Redis Cache

```powershell
docker exec -it redis redis-cli

# List keys
KEYS recommendation:*

# Get specific key
GET recommendation:feed:user123

# Clear cache
FLUSHDB
```

---

## 🚦 Status Check Command

Tạo alias để check nhanh:

```powershell
function Check-RecommendationServices {
    Write-Host "Checking services..." -ForegroundColor Cyan
    
    # Docker containers
    Write-Host "`nDocker Containers:" -ForegroundColor Yellow
    docker ps --format "table {{.Names}}\t{{.Status}}" | Select-String "recommendation|postgres|neo4j|redis|kafka"
    
    # Python service
    Write-Host "`nPython Service (8097):" -ForegroundColor Yellow
    try {
        $python = Invoke-RestMethod "http://localhost:8097/health"
        Write-Host "  Status: $($python.status)" -ForegroundColor Green
    } catch {
        Write-Host "  Status: DOWN" -ForegroundColor Red
    }
    
    # Java service
    Write-Host "`nJava Service (8095):" -ForegroundColor Yellow
    try {
        $java = Invoke-RestMethod "http://localhost:8095/actuator/health"
        Write-Host "  Status: $($java.status)" -ForegroundColor Green
    } catch {
        Write-Host "  Status: DOWN" -ForegroundColor Red
    }
}

# Sử dụng:
Check-RecommendationServices
```

---

## 📞 Cần trợ giúp?

1. Đọc **`RECOMMENDATION_DEV_SETUP_VN.md`** (hướng dẫn chi tiết)
2. Chạy **`.\test-recommendation-dev.ps1`** (test tự động)
3. Check logs trong terminal/IntelliJ
4. Check docker logs: `docker logs <container-name>`

---

**✨ TIP:** Bookmark các URLs sau để truy cập nhanh:
- Python Docs: http://localhost:8097/docs
- Java Health: http://localhost:8095/actuator/health
- Neo4j: http://localhost:7474

**🎉 Happy Coding!**
