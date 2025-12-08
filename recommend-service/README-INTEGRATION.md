# ✅ Recommendation Service - Eureka Integration & API Flow Setup

## 📋 Summary

**Vấn đề:** Recommend-service không đăng ký được với Eureka khi chạy local (dev mode).

**Nguyên nhân:** `eureka.client.enabled=false` trong `application-dev.yml`.

**Giải pháp:** Đã fix và bổ sung complete API flow từ client → gateway → recommend-service.

---

## 🔧 Thay đổi đã thực hiện

### 1. Fixed Eureka Registration ✅

**File:** `java-api/src/main/resources/application-dev.yml`

**Line 88-90:** Changed from `enabled: false` to `enabled: true`

```yaml
eureka:
  client:
    enabled: true  # ✅ Đã bật
    service-url:
      defaultZone: http://localhost:8761/eureka/
    register-with-eureka: true
    fetch-registry: true
```

### 2. Enhanced Debugging Logs ✅

**Files:**
- `controller/RecommendationController.java`
- `controller/FeedController.java`

**Thêm logs chi tiết:**
```java
log.info("========================================");
log.info("📥 API REQUEST: GET /api/recommend/posts");
log.info("   User ID: {}", userId);
log.info("   Page: {}, Size: {}", page, size);
log.info("========================================");
```

### 3. Updated API Gateway Routes ✅

**File:** `api-gateway/.../RouteConfig.java`

**Thêm routes:**
```java
.route("recommendation-api-route", r -> r
    .path("/api/recommend/**")
    .uri("lb://recommendation-service"))

.route("recommendation-feed-route", r -> r
    .path("/api/recommendation/**")
    .uri("lb://recommendation-service"))
```

### 4. Client-Frontend Integration ✅

**File:** `client-frontend/src/services/postService.ts`

**Thêm methods mới:**
- `getRecommendedPosts()` - Lấy gợi ý AI
- `getPersonalizedFeed()` - Lấy feed cá nhân hóa
- `recordRecommendationInteraction()` - Track tương tác
- `sendRecommendationFeedback()` - Gửi feedback

### 5. API Endpoints Constants ✅

**File:** `client-frontend/src/shared/constants/api-endpoints.ts`

**Thêm:**
```typescript
RECOMMENDATIONS: {
  BASE: '/api/recommend',
  POSTS: '/api/recommend/posts',
  FEED: '/api/recommendation/feed',
  // ...
}
```

---

## 🚀 Cách sử dụng

### Bước 1: Khởi động Recommendation Service

```bash
cd recommend-service/java-api

# Chạy bằng Maven
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# Hoặc chạy trong IDE (IntelliJ/Eclipse)
# Đảm bảo profile = "dev"
```

### Bước 2: Kiểm tra đăng ký Eureka

Mở browser: http://localhost:8761

Tìm `RECOMMENDATION-SERVICE` trong danh sách.

### Bước 3: Test API

```bash
# Chạy test script
cd recommend-service
.\test-recommendation-api.ps1

# Hoặc test thủ công
curl http://localhost:8095/api/recommend/health
```

### Bước 4: Sử dụng trong Frontend

```typescript
import { postService } from '@/services/postService';

// Lấy gợi ý cho user
const recommendations = await postService.getRecommendedPosts(userId, 0, 20);

// Hiển thị trong component
setPosts(recommendations);
```

---

## 📊 Luồng API hoàn chỉnh

```
┌──────────────┐
│ Client       │  GET /api/recommend/posts?userId=X
│ (React)      │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ API Gateway  │  Route: /api/recommend/** → recommendation-service
│ (Port 8090)  │
└──────┬───────┘
       │
       ↓
┌─────────────────────┐
│ Recommendation      │  Controller: RecommendationController
│ Service (Port 8095) │  Method: getRecommendedPosts()
└──────┬──────────────┘
       │
       ├─→ Neo4j (Graph data)
       ├─→ PostgreSQL (Post embeddings)
       ├─→ Redis (Cache)
       └─→ Python ML Service (Optional)
```

---

## 🎯 API Endpoints

### Recommend Service (AI-powered)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/recommend/posts` | Lấy gợi ý AI |
| GET | `/api/recommendation/feed` | Feed cá nhân hóa |
| POST | `/api/recommendation/interaction` | Ghi nhận tương tác |
| POST | `/api/recommend/feedback` | Gửi feedback |
| DELETE | `/api/recommend/cache/{userId}` | Xóa cache |
| GET | `/api/recommend/health` | Health check |

### Post Service (Legacy - Simple)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/recommendations/personalized/{userId}` | Gợi ý đơn giản |
| GET | `/api/recommendations/trending` | Bài viết trending |
| GET | `/api/posts/feed` | News feed truyền thống |

---

## 🐛 Debugging

### Xem Logs

Service sẽ log chi tiết mọi API call:

```
========================================
📥 API REQUEST: GET /api/recommend/posts
   User ID: user123
   Page: 0, Size: 20
========================================
🔄 Processing recommendation request...
========================================
📤 API RESPONSE: Success
   Total Recommendations: 20
   Algorithm: HYBRID_ML
   Generated At: 2024-12-08T14:00:00
========================================
```

### Các vấn đề thường gặp

#### 1. Service không đăng ký với Eureka

**Kiểm tra:**
```yaml
# application-dev.yml
eureka:
  client:
    enabled: true  # ← Phải là true
```

**Restart service sau khi thay đổi.**

#### 2. 404 Not Found

**Kiểm tra:**
- Eureka có service không? (http://localhost:8761)
- API Gateway routing đúng không?
- Endpoint path đúng không?

#### 3. Không có kết quả

**Kiểm tra:**
- Neo4j có dữ liệu không?
- PostgreSQL có posts không?
- Redis accessible không?

---

## 📚 Documentation Files

1. **QUICK-START.md** - Hướng dẫn nhanh
2. **API-FLOW-DOCUMENTATION.md** - Chi tiết API flow
3. **CHANGES-SUMMARY.md** - Tổng hợp thay đổi
4. **test-recommendation-api.ps1** - Script test

---

## ✅ Checklist trước khi test

- [ ] Eureka Server running (port 8761)
- [ ] API Gateway running (port 8090)
- [ ] Neo4j running (port 7687)
- [ ] PostgreSQL recommend_db (port 5435)
- [ ] Redis recommend-redis (port 6380)
- [ ] MongoDB post_db (port 27018)
- [ ] **Recommendation Service running (port 8095)** ← Start this!
- [ ] Service registered in Eureka (check dashboard)

---

## 🎓 Example Usage

### Homepage Component

```typescript
import { useEffect, useState } from 'react';
import { postService } from '@/services/postService';
import { useAuth } from '@/contexts/AuthContext';

export const HomePage = () => {
  const { user } = useAuth();
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    const loadFeed = async () => {
      try {
        // Lấy feed cá nhân hóa từ AI
        const feed = await postService.getPersonalizedFeed(user.id, 0, 20);
        setPosts(feed);
        
        // Track mỗi khi user xem post
        feed.forEach(post => {
          postService.recordRecommendationInteraction(
            user.id,
            post.id,
            'VIEW'
          );
        });
      } catch (error) {
        console.error('Failed to load feed:', error);
      }
    };

    if (user?.id) {
      loadFeed();
    }
  }, [user?.id]);

  return (
    <div>
      {posts.map(post => (
        <PostCard key={post.id} post={post} />
      ))}
    </div>
  );
};
```

---

## 🔄 Next Steps

1. **Start Service** - Chạy recommendation service với profile dev
2. **Verify** - Check Eureka dashboard
3. **Test** - Run test script hoặc curl
4. **Integrate** - Update frontend components để dùng API mới
5. **Monitor** - Theo dõi logs và performance

---

## 📞 Support

Nếu gặp vấn đề:

1. Check logs của recommendation service
2. Verify Eureka registration
3. Test endpoint directly (bypass gateway)
4. Check database connections
5. Review documentation files

---

**Status:** ✅ Setup Complete - Ready to Start Service

**Date:** December 8, 2024
