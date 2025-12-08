# REFACTORING PLAN - Recommend Service Optimization

## Vấn đề hiện tại

### Java API
1. **Duplicate Controllers**: 
   - `FeedController.java` (sử dụng `/api/recommendation/feed`)
   - `RecommendationController.java` (sử dụng `/api/recommend/posts`)
   - **Giải pháp**: Merge vào 1 controller duy nhất

2. **Duplicate Services**:
   - `RecommendationService.java` (interface)
   - `RecommendationServiceImpl.java` (implementation)
   - `HybridRecommendationService.java` (another implementation)
   - **Giải pháp**: Giữ 1 service duy nhất `RecommendationService`

3. **Duplicate Clients cho User Service**:
   - `UserServiceClient.java` (sử dụng WebClient)
   - `UserServiceFeignClient.java` (sử dụng Feign)
   - **Giải pháp**: Chỉ giữ Feign client (chuẩn microservice)

4. **NLP Logic không đúng chỗ**:
   - `AcademicClassifier.java` - Logic NLP ở Java
   - `EmbeddingService.java` - Embedding ở Java
   - **Giải pháp**: Remove, delegate hoàn toàn cho Python

5. **Duplicate Kafka Producers**:
   - `TrainingDataProducer.java`
   - `UserInteractionProducer.java`
   - **Giải pháp**: Merge thành 1 producer

### Python Model
1. **Duplicate Entry Points**:
   - `app.py` - FastAPI app với routes riêng
   - `server.py` - FastAPI app khác với embedding routes
   - **Giải pháp**: Chỉ giữ 1 entry point

2. **Chức năng chưa rõ ràng**:
   - `server.py`: Chỉ embedding
   - `app.py`: Có prediction và classification
   - **Giải pháp**: Merge tất cả routes vào 1 app

## Kiến trúc mục tiêu (theo ARCHITECTURE.md)

### Java Service (Port 8081)
- **Vai trò**: Orchestrator & Business Logic
- **Chức năng**:
  - REST API cho frontend
  - Gọi Python service để lấy embeddings/predictions
  - Business rules (filtering, boosting)
  - Caching (Redis)
  - Event streaming (Kafka)
  - DB operations (PostgreSQL, Neo4j)

### Python Service (Port 8000)
- **Vai trò**: AI Inference Engine ONLY
- **Chức năng**:
  - PhoBERT model inference
  - Generate embeddings (post, user)
  - Compute similarity
  - Academic classification
  - Prediction/Ranking

## Các bước refactoring

### Phase 1: Python Service - Consolidate Entry Points
1. ✅ Giữ `server.py` làm main entry point
2. ✅ Merge routes từ `app.py` vào `api/routes.py`
3. ✅ Remove `app.py`
4. ✅ Đảm bảo có đầy đủ endpoints theo ARCHITECTURE.md

### Phase 2: Java Service - Consolidate Controllers
1. ✅ Merge `FeedController` và `RecommendationController`
2. ✅ Unified endpoints:
   - `GET /api/recommendations/feed` - Main feed
   - `POST /api/recommendations/feedback` - Record interaction
   - `POST /api/recommendations/refresh` - Refresh cache
   - `GET /api/recommendations/health` - Health check

### Phase 3: Java Service - Consolidate Services
1. ✅ Remove `RecommendationServiceImpl` (không sử dụng)
2. ✅ Rename `HybridRecommendationService` → `RecommendationService`
3. ✅ Remove interface `RecommendationService` (không cần thiết cho single impl)

### Phase 4: Java Service - Consolidate Clients
1. ✅ Remove `UserServiceClient` (WebClient)
2. ✅ Giữ only `UserServiceFeignClient` + fallback
3. ✅ Giữ `PostServiceClient` (Feign)
4. ✅ Giữ `PythonModelServiceClient`

### Phase 5: Java Service - Remove Duplicate NLP Logic
1. ✅ Remove `AcademicClassifier.java`
2. ✅ Remove `EmbeddingService.java`
3. ✅ Remove `RankingEngine.java` (logic moved to service)
4. ✅ Delegate tất cả AI/NLP cho Python service

### Phase 6: Java Service - Consolidate Kafka Producers
1. ✅ Merge `TrainingDataProducer` và `UserInteractionProducer`
2. ✅ Single Kafka producer với methods riêng cho từng event type

## Kết quả mong đợi

### Cấu trúc Java API sau khi tối ưu:
```
java-api/
├── controller/
│   └── RecommendationController.java (ONLY ONE)
├── service/
│   └── RecommendationService.java (ONLY ONE)
├── client/
│   ├── PythonModelServiceClient.java
│   ├── PostServiceClient.java (Feign)
│   └── UserServiceClient.java (Feign)
├── kafka/
│   ├── consumer/ (unchanged)
│   └── producer/
│       └── EventProducer.java (UNIFIED)
├── repository/ (unchanged)
├── model/ (unchanged)
└── config/ (unchanged)
```

### Cấu trúc Python Model sau khi tối ưu:
```
python-model/
├── server.py (ONLY ENTRY POINT)
├── inference.py (PhoBERT engine)
├── api/
│   └── routes.py (ALL ROUTES)
├── services/
│   └── inference_service.py (RENAMED from prediction)
├── utils/
│   ├── similarity.py
│   └── feature_engineering.py
└── models/
    └── schemas.py
```

## Unified Communication Pattern

### Java → Python
- Tất cả gọi qua `PythonModelServiceClient`
- Endpoints:
  - `POST /embed/post` - Single post embedding
  - `POST /embed/post/batch` - Batch posts
  - `POST /embed/user` - User embedding
  - `POST /similarity/batch` - Batch similarity
  - `POST /api/model/predict` - ML ranking
  - `POST /api/model/classify/academic` - Classification

### Java → Other Services
- Tất cả dùng Feign Client (service discovery)
- `PostServiceClient` → post-service
- `UserServiceClient` → user-service

## Benefits

1. **Giảm code dư thừa**: ~30% code reduction
2. **Rõ ràng responsibility**: Java = orchestration, Python = AI
3. **Dễ maintain**: Single source of truth cho mỗi chức năng
4. **Dễ scale**: Clear boundaries giữa services
5. **Consistent**: Unified communication pattern
