# Recommendation Service Restructure

## Summary
Đã tái cấu trúc lại thư mục recommendation services theo kiến trúc chuẩn:

### Changes Made

#### 1. **recommendation-service** (Java Spring Boot)
- **Trước:** `recommendation-service-java/`
- **Sau:** `recommendation-service/`
- Cấu trúc:
  ```
  recommendation-service/
  ├── src/
  │   ├── main/
  │   │   ├── java/vn/ctu/edu/recommend/
  │   │   │   ├── client/          # Gọi Python NLP service
  │   │   │   ├── controller/      # REST API endpoints
  │   │   │   ├── service/         # Business logic, cache, filter
  │   │   │   ├── model/           # DTOs, Entities
  │   │   │   ├── config/          # Spring configurations
  │   │   │   ├── kafka/           # Kafka consumers/producers
  │   │   │   ├── ranking/         # Ranking algorithms
  │   │   │   ├── repository/      # Data access layer
  │   │   │   └── util/            # Utility classes
  │   │   └── resources/
  │   │       └── application.yml
  │   └── test/                    # Unit & Integration tests
  ├── Dockerfile
  └── pom.xml
  ```

#### 2. **nlp-service** (Python PhoBERT Service)
- **Trước:** `recommendation-service-python/`
- **Sau:** `nlp-service/`
- Cấu trúc đã được tái tổ chức hoàn toàn:
  ```
  nlp-service/
  ├── app/
  │   ├── main.py                  # FastAPI entrypoint (trước là app.py)
  │   ├── config.py                # Configuration settings
  │   ├── api/
  │   │   ├── __init__.py
  │   │   └── routes.py            # API routes
  │   ├── models/
  │   │   └── academic_model/      # PhoBERT model weights & tokenizer
  │   │       ├── pytorch_model.bin      (540 MB)
  │   │       ├── config.json
  │   │       ├── tokenizer_config.json
  │   │       ├── bpe.codes
  │   │       ├── vocab.txt
  │   │       └── special_tokens_map.json
  │   ├── services/                # Inference, ranking logic
  │   │   ├── __init__.py
  │   │   └── prediction_service.py
  │   └── utils/                   # Helper functions
  │       ├── __init__.py
  │       ├── feature_engineering.py
  │       └── similarity.py
  ├── requirements.txt
  ├── Dockerfile                   # Updated CMD for uvicorn
  └── .env.example
  ```

### Key Updates

#### nlp-service/Dockerfile
```dockerfile
# OLD
CMD ["python", "app.py"]

# NEW  
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8097"]
```

#### nlp-service/app/main.py
- Entry point đã được chuẩn hóa
- Import paths phù hợp với cấu trúc mới
- FastAPI application initialization

### Benefits of New Structure

1. **Separation of Concerns**
   - Java service tập trung vào API, caching, filtering
   - Python service chỉ xử lý ML/NLP tasks

2. **Standard Package Layout**
   - Tuân theo chuẩn Java Maven project
   - Tuân theo chuẩn Python application structure

3. **Easier Maintenance**
   - Rõ ràng về responsibility của từng service
   - Dễ scale và deploy độc lập

4. **Better Organization**
   - Models được tách riêng trong `app/models/`
   - Business logic trong `app/services/`
   - Utilities tập trung trong `app/utils/`

### Migration Notes

#### Old Structure (còn tồn tại)
- `recommendation-service-java/` - có thể xóa sau khi verify
- `recommendation-service-python/` - có thể xóa sau khi verify

#### New Structure (active)
- `recommendation-service/` - Java Spring Boot service
- `nlp-service/` - Python ML/NLP service

### Next Steps

1. **Update docker-compose.yml**
   - Thay đổi service names
   - Update volume mounts
   - Update environment variables

2. **Update Service Discovery**
   - Eureka registration với tên mới
   - API Gateway routing updates

3. **Update Documentation**
   - README files
   - API documentation
   - Deployment guides

4. **Testing**
   - Verify Java service builds
   - Verify Python service runs
   - Test integration between services

5. **Cleanup** (sau khi verify)
   ```powershell
   # Remove old directories
   Remove-Item -Path "recommendation-service-java" -Recurse -Force
   Remove-Item -Path "recommendation-service-python" -Recurse -Force
   ```

### Docker Compose Updates Needed

```yaml
services:
  recommendation-service:  # Renamed from recommendation-service-java
    build: ./recommendation-service
    # ... rest of config
    
  nlp-service:  # Renamed from recommendation-service-python
    build: ./nlp-service
    command: python -m uvicorn app.main:app --host 0.0.0.0 --port 8097
    # ... rest of config
```

### Verification Commands

```powershell
# Check Java service structure
Get-ChildItem ".\recommendation-service\src\main\java\vn\ctu\edu\recommend" -Directory

# Check Python service structure  
Get-ChildItem ".\nlp-service\app" -Directory

# Verify model files exist
Get-ChildItem ".\nlp-service\app\models\academic_model" | Select Name, Length

# Test Java build (if Maven installed)
cd recommendation-service
mvn clean compile

# Test Python dependencies
cd nlp-service
pip install -r requirements.txt
```

## Status
✅ Directory restructure completed
⚠️ Requires docker-compose.yml updates
⚠️ Requires testing before removing old directories
