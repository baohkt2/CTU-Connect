# User Service Endpoints - Updated & Aligned with Frontend

## ✅ Hoàn thành tất cả endpoints cần thiết

### 📋 Danh sách Endpoints đã cập nhật

#### 1. Profile Management
| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `GET` | `/api/users/profile` | Get current user profile | ✅ NEW |
| `GET` | `/api/users/me/profile` | Get current user profile (alt) | ✅ UPDATED |
| `PUT` | `/api/users/profile` | Update current user profile | ✅ NEW |
| `GET` | `/api/users/:id` | Get user by ID | ✅ EXISTS |

#### 2. Friend Management
| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `POST` | `/api/users/:id/friend-request` | Send friend request | ✅ UPDATED |
| `POST` | `/api/users/:id/accept-friend` | Accept friend request | ✅ UPDATED |
| `POST` | `/api/users/:id/reject-friend` | Reject friend request | ✅ NEW |
| `DELETE` | `/api/users/:id/friend` | Remove friend/Unfriend | ✅ NEW |
| `DELETE` | `/api/users/:id/friend-request` | Cancel sent friend request | ✅ NEW |
| `GET` | `/api/users/sent-requests` | Get sent friend requests | ✅ NEW |
| `GET` | `/api/users/received-requests` | Get received friend requests | ✅ NEW |
| `GET` | `/api/users/:id/friends` | Get user's friends | ✅ EXISTS |

#### 3. Friend Suggestions & Mutual Friends
| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `GET` | `/api/users/friend-suggestions` | Get friend suggestions | ✅ UPDATED |
| `GET` | `/api/users/:id/mutual-friends-count` | Get mutual friends count | ✅ UPDATED |

#### 4. Search & Discovery
| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `GET` | `/api/users/search` | Search users with filters | ✅ UPDATED |

#### 5. Timeline & Activities
| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `GET` | `/api/users/:id/timeline` | Get user timeline (redirects to post-service) | ✅ NEW |
| `GET` | `/api/users/:id/activities` | Get user activity feed | ✅ UPDATED |

#### 6. Internal Service Endpoints (for microservices)
| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `GET` | `/api/users/:id/friends/ids` | Get friend IDs | ✅ EXISTS |
| `GET` | `/api/users/:id/close-interactions` | Get close interaction IDs | ✅ EXISTS |
| `GET` | `/api/users/:id/same-faculty` | Get same faculty user IDs | ✅ EXISTS |
| `GET` | `/api/users/:id/same-major` | Get same major user IDs | ✅ EXISTS |
| `GET` | `/api/users/:id/interest-tags` | Get user interest tags | ✅ EXISTS |
| `GET` | `/api/users/:id/preferred-categories` | Get preferred categories | ✅ EXISTS |
| `GET` | `/api/users/:id/faculty-id` | Get user's faculty ID | ✅ EXISTS |
| `GET` | `/api/users/:id/major-id` | Get user's major ID | ✅ EXISTS |

---

## 🔧 Những thay đổi quan trọng

### 1. **Profile Endpoints**
- ✅ Thêm `GET /api/users/profile` làm alias cho `/me/profile`
- ✅ Thêm `PUT /api/users/profile` để update profile của current user
- ✅ Tất cả đều dùng `SecurityContextHolder` để lấy authenticated user

### 2. **Friend Request Simplification**
- ✅ Đơn giản hóa API để frontend dễ dùng:
  - `POST /api/users/:targetUserId/friend-request` - gửi request
  - `POST /api/users/:requesterId/accept-friend` - chấp nhận
  - `POST /api/users/:requesterId/reject-friend` - từ chối
  - `DELETE /api/users/:friendId/friend` - unfriend
- ✅ Tự động invalidate cache sau mọi thao tác

### 3. **Request Management**
- ✅ Thêm endpoints để xem sent/received requests:
  - `GET /api/users/sent-requests`
  - `GET /api/users/received-requests`
- ✅ Cho phép cancel request đã gửi

### 4. **Authentication Consistency**
- ✅ Tất cả endpoints đều dùng `@RequireAuth`
- ✅ Lấy current user từ `SecurityContextHolder.getAuthenticatedUser()`
- ✅ Không còn phụ thuộc vào method parameter injection

### 5. **Timeline Endpoint**
- ℹ️ `/api/users/:id/timeline` trả về message redirect đến post-service
- ℹ️ Timeline posts nên query từ `/api/posts/timeline/:userId`

---

## 📝 Cách sử dụng từ Frontend

### Get Current User Profile
```typescript
// GET /api/users/profile
const response = await fetch('/api/users/profile', {
  headers: { 'Authorization': `Bearer ${token}` }
});
const profile = await response.json();
```

### Update Profile
```typescript
// PUT /api/users/profile
await fetch('/api/users/profile', {
  method: 'PUT',
  headers: { 
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(updateData)
});
```

### Send Friend Request
```typescript
// POST /api/users/:id/friend-request
await fetch(`/api/users/${targetUserId}/friend-request`, {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${token}` }
});
```

### Accept Friend Request
```typescript
// POST /api/users/:id/accept-friend
await fetch(`/api/users/${requesterId}/accept-friend`, {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${token}` }
});
```

### Search Users
```typescript
// GET /api/users/search?query=john&faculty=IT&page=0&size=20
const response = await fetch('/api/users/search?query=john', {
  headers: { 'Authorization': `Bearer ${token}` }
});
const users = await response.json();
```

---

## 🚀 Deployment

Rebuild và restart user-service:
```bash
cd user-service
mvn clean package -DskipTests
docker-compose restart user-service
```

---

## ✨ Summary

**Tổng số endpoints:** 30+ endpoints
- **Profile:** 4 endpoints
- **Friend Management:** 8 endpoints
- **Search & Discovery:** 1 endpoint
- **Timeline & Activities:** 2 endpoints
- **Friend Suggestions:** 2 endpoints
- **Internal Services:** 8+ endpoints

**Tất cả đã align với frontend requirements!** 🎉
