# Friend Feature API Usage Guide

## Hướng dẫn sử dụng các API tính năng bạn bè

### Authentication
Tất cả các API đều yêu cầu JWT token trong header:
```
Authorization: Bearer <your-jwt-token>
```

---

## 1. Quản lý danh sách bạn bè

### Lấy danh sách bạn bè của mình
```http
GET /api/users/me/friends?page=0&size=20
```

**Response:**
```json
{
  "content": [
    {
      "id": "user-123",
      "fullName": "Nguyễn Văn A",
      "username": "nguyenvana",
      "email": "nguyenvana@student.ctu.edu.vn",
      "avatarUrl": "https://...",
      "faculty": "Công nghệ thông tin",
      "major": "Khoa học máy tính",
      "batch": "2020",
      "friendsCount": 150,
      "mutualFriendsCount": 25
    }
  ],
  "totalElements": 100,
  "totalPages": 5,
  "number": 0,
  "size": 20
}
```

---

## 2. Quản lý lời mời kết bạn

### Lấy lời mời kết bạn nhận được
```http
GET /api/users/me/friend-requests
```

**Response:**
```json
[
  {
    "id": "user-456",
    "fullName": "Trần Thị B",
    "username": "tranthib",
    "email": "tranthib@student.ctu.edu.vn",
    "faculty": "Kinh tế",
    "major": "Quản trị kinh doanh",
    "mutualFriendsCount": 5,
    "requestType": "RECEIVED"
  }
]
```

### Lấy lời mời kết bạn đã gửi
```http
GET /api/users/me/friend-requested
```

**Response:** Tương tự như trên với `requestType: "SENT"`

### Gửi lời mời kết bạn
```http
POST /api/users/me/invite/{friendId}
```

**Response:**
```json
{
  "message": "Friend request sent successfully"
}
```

### Chấp nhận lời mời kết bạn
```http
POST /api/users/me/accept-invite/{friendId}
```

**Response:**
```json
{
  "message": "Friend request accepted successfully"
}
```

### Từ chối lời mời kết bạn
```http
POST /api/users/me/reject-invite/{friendId}
```

**Response:**
```json
{
  "message": "Friend request rejected successfully"
}
```

### Hủy kết bạn
```http
DELETE /api/users/me/friends/{friendId}
```

**Response:**
```json
{
  "message": "Friend removed successfully"
}
```

---

## 3. Gợi ý kết bạn

### Lấy gợi ý kết bạn thông minh
```http
GET /api/users/friend-suggestions?limit=20
```

Sử dụng thuật toán thông minh dựa trên:
- Bạn chung (mutual friends)
- Cùng khoa, ngành, niên khóa
- Sở thích tương tự

**Response:**
```json
[
  {
    "userId": "user-789",
    "fullName": "Lê Văn C",
    "username": "levanc",
    "avatarUrl": "https://...",
    "mutualFriendsCount": 10,
    "sameFaculty": true,
    "sameBatch": true,
    "relevanceScore": 0.85,
    "reason": "10 mutual friends, same faculty and batch"
  }
]
```

### Tìm kiếm người dùng với bộ lọc

#### Tìm theo tên
```http
GET /api/users/friend-suggestions/search?query=nguyen&limit=50
```

#### Lọc theo khoa
```http
GET /api/users/friend-suggestions/search?faculty=Công nghệ thông tin&limit=50
```

#### Lọc theo niên khóa
```http
GET /api/users/friend-suggestions/search?batch=2020&limit=50
```

#### Kết hợp nhiều bộ lọc
```http
GET /api/users/friend-suggestions/search?query=nguyen&faculty=Công nghệ thông tin&batch=2020&limit=50
```

**Response:**
```json
[
  {
    "id": "user-101",
    "fullName": "Nguyễn Thị D",
    "username": "nguyenthid",
    "email": "nguyenthid@student.ctu.edu.vn",
    "faculty": "Công nghệ thông tin",
    "major": "Công nghệ phần mềm",
    "batch": "2020",
    "friendsCount": 80,
    "mutualFriendsCount": 3,
    "isFriend": false,
    "requestSent": false,
    "requestReceived": false,
    "sameFaculty": true,
    "sameBatch": true
  }
]
```

**Lưu ý:**
- Kết quả tự động lọc bỏ người dùng đã là bạn bè
- Kết quả tự động lọc bỏ người dùng hiện tại
- Nếu `query` null: chỉ áp dụng bộ lọc (college, faculty, batch)
- Nếu `query` có giá trị: tìm theo tên/email và áp dụng bộ lọc

---

## 4. Kiểm tra trạng thái quan hệ

### Lấy trạng thái quan hệ với một user
```http
GET /api/users/{targetUserId}/friendship-status
```

**Response:**
```json
{
  "status": "none"  // Có thể là: "none", "friends", "sent", "received", "self"
}
```

**Các trạng thái:**
- `none`: Chưa có quan hệ
- `friends`: Đã là bạn bè
- `sent`: Đã gửi lời mời (đang chờ)
- `received`: Đã nhận lời mời (cần chấp nhận/từ chối)
- `self`: Đang xem profile của chính mình

**Use case:** Frontend có thể dùng để hiển thị button phù hợp:
- `none` → Hiển thị "Kết bạn"
- `friends` → Hiển thị "Bạn bè"
- `sent` → Hiển thị "Hủy lời mời"
- `received` → Hiển thị "Chấp nhận" / "Từ chối"
- `self` → Không hiển thị button

---

## 5. Bạn chung

### Lấy số lượng bạn chung
```http
GET /api/users/{targetUserId}/mutual-friends-count
```

**Response:**
```json
{
  "count": 15
}
```

### Lấy danh sách bạn chung
```http
GET /api/users/{targetUserId}/mutual-friends?page=0&size=20
```

**Response:**
```json
{
  "content": [
    {
      "id": "user-202",
      "fullName": "Phạm Văn E",
      "username": "phamvane",
      "avatarUrl": "https://...",
      "faculty": "Công nghệ thông tin"
    }
  ],
  "totalElements": 15,
  "totalPages": 1,
  "number": 0,
  "size": 20
}
```

---

## 6. Các API thay thế (Alternative paths)

Nếu bạn muốn sử dụng path khác, các API dưới đây cũng hoạt động:

```http
POST /api/users/{targetUserId}/friend-request        # Thay vì /me/invite/{targetUserId}
POST /api/users/{requesterId}/accept-friend           # Thay vì /me/accept-invite/{requesterId}
POST /api/users/{requesterId}/reject-friend           # Thay vì /me/reject-invite/{requesterId}
DELETE /api/users/{friendId}/friend                   # Thay vì /me/friends/{friendId}
GET /api/users/sent-requests                          # Thay vì /me/friend-requested
GET /api/users/received-requests                      # Thay vì /me/friend-requests
```

---

## 7. Error Handling

### Common Error Responses

**401 Unauthorized:**
```json
{
  "error": "Unauthorized",
  "message": "No authenticated user found"
}
```

**404 Not Found:**
```json
{
  "error": "Not Found",
  "message": "User not found with ID: xxx"
}
```

**400 Bad Request:**
```json
{
  "error": "Bad Request",
  "message": "Cannot send friend request to yourself"
}
```

**409 Conflict:**
```json
{
  "error": "Conflict",
  "message": "Unable to send friend request. Users may already be friends or request already exists"
}
```

---

## 8. Frontend Integration Example

### React/TypeScript Example

```typescript
import { userService } from '@/services/userService';

// Get my friends
const friends = await userService.getMyFriends();

// Search for users by faculty
const users = await userService.searchFriendSuggestions({
  faculty: 'Công nghệ thông tin',
  limit: 50
});

// Send friend request
await userService.sendFriendRequest(targetUserId);

// Check friendship status
const status = await userService.getFriendshipStatus(targetUserId);

// Get mutual friends
const mutualFriends = await userService.getMutualFriendsWithUser(targetUserId);
```

### Using FriendButton Component

```tsx
import { FriendButton } from '@/components/ui/FriendButton';

<FriendButton
  targetUserId={user.id}
  initialStatus="none"
  onStatusChange={(newStatus) => {
    console.log('Status changed to:', newStatus);
  }}
  size="md"
/>
```

Component tự động handle:
- Gửi lời mời kết bạn
- Chấp nhận/từ chối lời mời
- Hủy kết bạn
- Hiển thị button phù hợp với từng trạng thái

---

## 9. Testing

### Using PowerShell Script
```powershell
# Set your JWT token
$TOKEN = "your-jwt-token-here"

# Run the test script
.\test-friend-api.ps1
```

### Using curl

```bash
# Get my friends
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/users/me/friends

# Search users by faculty
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8080/api/users/friend-suggestions/search?faculty=Công nghệ thông tin"

# Send friend request
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/users/me/invite/user-123

# Get friendship status
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/users/user-123/friendship-status
```

---

## 10. Best Practices

1. **Pagination**: Luôn sử dụng pagination cho danh sách lớn
   ```
   ?page=0&size=20
   ```

2. **Caching**: Frontend nên cache kết quả friendship status và mutual friends count

3. **Real-time Updates**: Cân nhắc sử dụng WebSocket để cập nhật real-time khi có lời mời kết bạn mới

4. **Error Handling**: Luôn handle errors và hiển thị thông báo thân thiện với user

5. **Loading States**: Hiển thị loading spinner khi đang call API

6. **Debouncing**: Sử dụng debounce cho search input (300-500ms)

---

## 11. Performance Tips

1. **Lazy Loading**: Load friends list và suggestions khi cần thiết, không load tất cả ngay từ đầu

2. **Infinite Scroll**: Sử dụng infinite scroll cho danh sách dài thay vì pagination buttons

3. **Batch Requests**: Khi cần lấy friendship status của nhiều users, cân nhắc tạo batch API

4. **Cache Invalidation**: Invalidate cache khi:
   - Gửi/chấp nhận/từ chối friend request
   - Unfriend someone
   - User updates profile

---

**Last Updated**: December 9, 2025
