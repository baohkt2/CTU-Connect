# Chat Feature - Quick Start Guide

## Khởi động Chat Feature

### Bước 1: Start Chat Database
```powershell
.\start-chat-db.ps1
```

Hoặc thủ công:
```powershell
docker-compose up -d chat_db
```

Kiểm tra:
- MongoDB chạy trên `localhost:27019`
- Database name: `chat_db`

### Bước 2: Start Chat Service trong IDE

#### IntelliJ IDEA / Eclipse
1. Mở project `chat-service`
2. Tìm file `ChatServiceApplication.java`
3. Right-click → Run 'ChatServiceApplication'

#### Command Line
```powershell
cd chat-service
.\mvnw.cmd spring-boot:run
```

Kiểm tra:
- Service chạy trên `http://localhost:8086`
- Đăng ký với Eureka Server (localhost:8761)

### Bước 3: Start Frontend
```powershell
cd client-frontend
npm run dev
```

Frontend chạy trên `http://localhost:3000`

## Kiểm tra Chat Feature

### 1. Test qua UI

#### A. Từ Friends List
1. Login vào hệ thống
2. Vào trang Friends (`/friends`)
3. Thấy nút màu xanh lá "Nhắn tin" bên cạnh mỗi bạn bè
4. Click "Nhắn tin" → Chuyển đến `/messages?userId={friendId}`
5. Conversation tự động được tạo và hiển thị

#### B. Gửi Message
1. Trong chat window, nhập text vào input box
2. Press Enter hoặc click nút Send (icon paper plane)
3. Message xuất hiện trong chat với:
   - Avatar và tên người gửi (nếu là người khác)
   - Bubble màu xanh (own) hoặc xám (others)
   - Thời gian gửi

#### C. Gửi File/Image
1. Click nút attachment (icon paperclip)
2. Chọn file (hỗ trợ: images, PDF, DOC, TXT)
3. File upload lên media-service
4. Message xuất hiện với:
   - Image: hiển thị inline, click để xem full
   - File: hiển thị link với icon, click để download

### 2. Test qua API

#### Tạo Direct Conversation
```bash
curl -X POST http://localhost:8090/api/chats/conversations/direct/{friendId} \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json"
```

Response:
```json
{
  "id": "conv_123",
  "type": "DIRECT",
  "participants": [...],
  "createdAt": "2024-12-10T07:00:00"
}
```

#### Gửi Text Message
```bash
curl -X POST http://localhost:8090/api/chats/messages \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "conversationId": "conv_123",
    "content": "Hello, this is a test message"
  }'
```

#### Gửi Message với Media
```bash
# Bước 1: Upload file
curl -X POST http://localhost:8090/api/media/upload \
  -H "Authorization: Bearer {token}" \
  -F "file=@/path/to/image.jpg" \
  -F "type=image"

# Response: { "url": "https://...", "thumbnailUrl": "https://..." }

# Bước 2: Gửi message với attachment
curl -X POST http://localhost:8090/api/chats/messages \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "conversationId": "conv_123",
    "content": "Check out this image",
    "attachment": {
      "fileName": "image.jpg",
      "fileUrl": "https://...",
      "fileType": "image/jpeg",
      "fileSize": 123456,
      "thumbnailUrl": "https://..."
    }
  }'
```

#### Lấy Messages
```bash
curl -X GET "http://localhost:8090/api/chats/messages/conversation/{conversationId}?page=0&size=50" \
  -H "Authorization: Bearer {token}"
```

## Troubleshooting

### Issue: Chat DB không start
```powershell
# Kiểm tra Docker
docker ps | findstr chat

# Xem logs
docker-compose logs chat_db

# Restart
docker-compose restart chat_db
```

### Issue: Chat Service không connect được MongoDB
Check `application.properties`:
```properties
spring.data.mongodb.host=localhost
spring.data.mongodb.port=27019
spring.data.mongodb.database=chat_db
```

### Issue: Không tạo được conversation
- Kiểm tra user-service đang chạy (cần để lấy user info)
- Kiểm tra JWT token hợp lệ
- Xem logs chat-service

### Issue: Upload file không thành công
- Kiểm tra media-service đang chạy
- Kiểm tra file size không quá lớn
- Xem logs media-service

### Issue: Message không hiển thị trong UI
- F12 → Network tab → Kiểm tra API calls
- Console → Xem errors
- Kiểm tra conversationId đúng

## Architecture Overview

```
User clicks "Nhắn tin"
    ↓
Navigate to /messages?userId={friendId}
    ↓
ChatSidebar → POST /api/chats/conversations/direct/{friendId}
    ↓
API Gateway (JWT validation)
    ↓
Chat Service → ConversationService.getOrCreateDirectConversation()
    ├→ Check if exists in MongoDB
    ├→ Create new if not exists
    └→ Return conversation
    ↓
Conversation displayed and selected
    ↓
User types message
    ↓
ChatMessageArea → POST /api/chats/messages
    ↓
Chat Service → MessageService.sendMessage()
    ├→ Fetch user info from user-service
    ├→ Save message to MongoDB
    └→ Return message response
    ↓
Message displayed in chat
```

## File Upload Flow

```
User selects file
    ↓
ChatMessageArea → Upload to media-service
    ↓
POST /api/media/upload (FormData)
    ↓
Media Service
    ├→ Store file (local/S3)
    └→ Return { url, thumbnailUrl }
    ↓
Send message with attachment
    ↓
POST /api/chats/messages
{
  conversationId: "...",
  content: "filename",
  attachment: { fileUrl, fileName, fileType, ... }
}
    ↓
Message saved with attachment reference
    ↓
Display in chat:
  - Image: <img src={fileUrl} />
  - File: <a href={fileUrl}>Download</a>
```

## Database Schema

### Conversation Document
```javascript
{
  _id: "conv_123",
  type: "DIRECT", // or "GROUP"
  participantIds: ["user1", "user2"],
  name: null, // for GROUP only
  lastMessageId: "msg_456",
  lastMessageAt: ISODate("2024-12-10T07:00:00Z"),
  createdBy: "user1",
  createdAt: ISODate("2024-12-10T06:00:00Z"),
  updatedAt: ISODate("2024-12-10T07:00:00Z")
}
```

### Message Document
```javascript
{
  _id: "msg_456",
  conversationId: "conv_123",
  senderId: "user1",
  senderName: "Nguyen Van A",
  senderAvatar: "https://...",
  type: "TEXT", // or "IMAGE", "FILE"
  content: "Hello World",
  attachment: {
    fileName: "image.jpg",
    fileUrl: "https://...",
    fileType: "image/jpeg",
    fileSize: 123456,
    thumbnailUrl: "https://..."
  },
  replyToMessageId: null,
  reactions: [],
  status: "SENT", // or "DELIVERED", "READ"
  readByUserIds: ["user2"],
  createdAt: ISODate("2024-12-10T07:00:00Z"),
  updatedAt: ISODate("2024-12-10T07:00:00Z"),
  isEdited: false,
  isDeleted: false
}
```

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/chats/conversations` | Lấy danh sách conversations |
| POST | `/api/chats/conversations/direct/{friendId}` | Tạo/lấy conversation với friend |
| GET | `/api/chats/conversations/{id}` | Lấy chi tiết conversation |
| GET | `/api/chats/messages/conversation/{id}` | Lấy messages của conversation |
| POST | `/api/chats/messages` | Gửi message |
| PUT | `/api/chats/messages/{id}` | Sửa message |
| DELETE | `/api/chats/messages/{id}` | Xóa message |
| POST | `/api/chats/messages/conversation/{id}/mark-read` | Đánh dấu đã đọc |
| POST | `/api/media/upload` | Upload file/image |

## UI Components

### ChatSidebar
- **Responsibilities**: 
  - Hiển thị danh sách conversations
  - Search conversations
  - Tạo conversation mới
  - Select conversation
- **Props**: `selectedConversationId`, `onSelectConversation`, `friendUserId`
- **Location**: `src/components/chat/ChatSidebar.tsx`

### ChatMessageArea
- **Responsibilities**:
  - Hiển thị messages
  - Gửi text messages
  - Upload và gửi files/images
  - Auto-scroll
- **Props**: `conversationId`, `currentUserId`
- **Location**: `src/components/chat/ChatMessageArea.tsx`

## Next Development Steps

1. **WebSocket Real-time**: Implement STOMP client trong frontend
2. **Typing Indicators**: Hiển thị "đang nhập..." 
3. **Read Receipts**: Tick xanh khi đã đọc
4. **Online Status**: Cập nhật real-time
5. **Reactions**: Emoji reactions trên messages
6. **Message Actions**: Edit, Delete, Forward, Reply
7. **Group Chat**: Tạo và quản lý nhóm
8. **Voice Messages**: Record và gửi audio
9. **Video Calls**: WebRTC integration

## Support

Nếu gặp vấn đề:
1. Check logs: chat-service, media-service, MongoDB
2. Verify JWT token valid
3. Check network tab in browser DevTools
4. Review CHAT-FEATURE-IMPLEMENTATION.md
