# 💬 CTU Connect - Chat Feature

## ✨ Features

### Core Functionality
- ✅ **Real-time Chat**: Hệ thống chat trực tiếp giữa bạn bè
- ✅ **Text Messages**: Gửi và nhận tin nhắn văn bản
- ✅ **Media Support**: Gửi hình ảnh và files (PDF, DOC, TXT)
- ✅ **Direct Conversations**: Chat 1-1 với bạn bè
- ✅ **Conversation List**: Danh sách các cuộc trò chuyện với preview
- ✅ **Search**: Tìm kiếm conversations
- ✅ **Message History**: Lịch sử tin nhắn được lưu trữ

### UI/UX
- 🎨 **Messenger-inspired**: Giao diện giống Facebook Messenger
- 💬 **Bubble Chat**: Message bubbles với màu sắc phân biệt
- 👤 **Avatars**: Hiển thị avatar người dùng
- 🕐 **Timestamps**: Thời gian gửi tin nhắn động
- 📱 **Responsive**: Tương thích mọi kích thước màn hình
- 🔍 **Empty States**: Thông báo rõ ràng khi chưa có dữ liệu

### Technical
- 🔐 **Secure**: JWT authentication cho mọi request
- 💾 **MongoDB**: Lưu trữ messages và conversations
- 🚀 **Scalable**: Microservices architecture
- 📤 **File Upload**: Tích hợp media-service
- 👥 **User Info**: Lấy thông tin user từ user-service

## 🏗️ Architecture

### Backend Services
```
chat-service (Port 8086)
  ├─ REST API for conversations and messages
  ├─ WebSocket support for real-time (ready to implement)
  ├─ MongoDB for data persistence
  └─ Integration with user-service and media-service

chat_db (MongoDB)
  ├─ Port: 27019
  ├─ Database: chat_db
  └─ Collections: conversations, messages
```

### Frontend Components
```
/messages (Chat Page)
  ├─ ChatSidebar (Conversations list)
  └─ ChatMessageArea (Chat window)

/friends (Friends Page)
  └─ "Nhắn tin" button → Navigate to /messages
```

## 🚀 Quick Start

### 1. Start Database
```powershell
.\start-chat-db.ps1
```

### 2. Start Chat Service
**Option A: IDE (Recommended)**
- Open `chat-service` project
- Run `ChatServiceApplication.java`

**Option B: Command Line**
```powershell
cd chat-service
.\mvnw.cmd spring-boot:run
```

### 3. Start Frontend
```powershell
cd client-frontend
npm run dev
```

### 4. Test Chat
1. Login to the system
2. Go to Friends page
3. Click "Nhắn tin" on any friend
4. Start chatting!

## 📋 Prerequisites

### Required Services
- ✅ Eureka Server (8761)
- ✅ API Gateway (8090)
- ✅ User Service (8081)
- ✅ Media Service (8084)
- ✅ Auth Service (8080)
- ✅ Redis (6379)
- ✅ MongoDB (27019 for chat_db)

### All Running
```powershell
# Check all services
docker-compose ps

# Should show:
# - eureka-server
# - api-gateway
# - chat_db (MongoDB)
# - redis
# - neo4j (for user-service)
# - media_db (PostgreSQL)
```

## 📖 Documentation

### Detailed Guides
- 📘 [Implementation Summary](./CHAT-FEATURE-IMPLEMENTATION.md) - Chi tiết kỹ thuật
- 🚀 [Quick Start Guide](./CHAT-QUICK-START.md) - Hướng dẫn khởi động và test
- 🔧 [API Reference](#api-reference) - Danh sách endpoints

### API Reference

#### Conversations
```http
# Get conversations list
GET /api/chats/conversations?page=0&size=20
Authorization: Bearer {token}

# Create/Get direct conversation
POST /api/chats/conversations/direct/{friendId}
Authorization: Bearer {token}

# Get conversation details
GET /api/chats/conversations/{conversationId}
Authorization: Bearer {token}
```

#### Messages
```http
# Get messages
GET /api/chats/messages/conversation/{conversationId}?page=0&size=50
Authorization: Bearer {token}

# Send message
POST /api/chats/messages
Authorization: Bearer {token}
Content-Type: application/json

{
  "conversationId": "conv_123",
  "content": "Hello World",
  "attachment": {  // Optional
    "fileName": "image.jpg",
    "fileUrl": "https://...",
    "fileType": "image/jpeg",
    "fileSize": 123456,
    "thumbnailUrl": "https://..."
  }
}

# Mark as read
POST /api/chats/messages/conversation/{conversationId}/mark-read
Authorization: Bearer {token}
```

#### Media Upload
```http
POST /api/media/upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: [binary]
type: "image" | "document"
```

## 💡 Usage Examples

### Send Text Message
```javascript
const response = await apiClient.post('/api/chats/messages', {
  conversationId: 'conv_123',
  content: 'Hello, how are you?'
});
```

### Send Image
```javascript
// 1. Upload image
const formData = new FormData();
formData.append('file', imageFile);
formData.append('type', 'image');

const uploadRes = await apiClient.post('/api/media/upload', formData);

// 2. Send message with image
const messageRes = await apiClient.post('/api/chats/messages', {
  conversationId: 'conv_123',
  content: 'Check this out!',
  attachment: {
    fileName: imageFile.name,
    fileUrl: uploadRes.url,
    fileType: imageFile.type,
    fileSize: imageFile.size,
    thumbnailUrl: uploadRes.thumbnailUrl
  }
});
```

## 🎯 User Stories

### As a User
1. **Start Chat from Friends**
   - I can see a "Nhắn tin" button next to each friend
   - When I click it, I navigate to the chat page
   - The conversation is automatically created and opened

2. **Send Messages**
   - I can type text and press Enter to send
   - I can see my messages on the right (blue bubble)
   - I can see friend's messages on the left (gray bubble)
   - Messages show timestamp and sender name

3. **Send Media**
   - I can click the attachment icon to select files
   - I can send images (displayed inline)
   - I can send documents (displayed as download links)
   - Upload progress is shown

4. **View Conversations**
   - I can see all my conversations in the sidebar
   - Each shows: avatar, name, last message, time
   - I can search conversations
   - Unread count is shown if applicable

## 🔧 Configuration

### Chat Service (`application.properties`)
```properties
server.port=8086
spring.application.name=chat-service

# MongoDB
spring.data.mongodb.database=chat_db
spring.data.mongodb.host=localhost
spring.data.mongodb.port=27019

# Redis
spring.data.redis.host=localhost
spring.data.redis.port=6379

# Kafka
spring.kafka.bootstrap-servers=localhost:9092

# Eureka
eureka.client.service-url.defaultZone=http://localhost:8761/eureka/
```

### Docker Compose
```yaml
chat_db:
  image: mongo:7.0
  container_name: chat_mongodb
  ports:
    - "27019:27017"
  environment:
    - MONGO_INITDB_DATABASE=chat_db
  volumes:
    - chat-data:/data/db
```

## 🧪 Testing

### Manual Testing Flow
1. **Setup**: Start all required services
2. **Login**: Authenticate as user A
3. **Navigate**: Go to Friends page
4. **Action**: Click "Nhắn tin" on user B
5. **Verify**: 
   - Redirected to `/messages?userId={userB}`
   - Conversation created
   - Chat window shows empty state or history
6. **Send Text**: Type "Hello" and send
7. **Verify**: Message appears with correct styling
8. **Send Image**: Upload an image
9. **Verify**: Image displays inline
10. **Switch User**: Login as user B
11. **Verify**: See user A's messages

### API Testing with cURL
```bash
# Get JWT token first
TOKEN="your_jwt_token_here"

# Test create conversation
curl -X POST http://localhost:8090/api/chats/conversations/direct/user123 \
  -H "Authorization: Bearer $TOKEN"

# Test send message
curl -X POST http://localhost:8090/api/chats/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"conversationId":"conv123","content":"Test message"}'

# Test get messages
curl -X GET http://localhost:8090/api/chats/messages/conversation/conv123 \
  -H "Authorization: Bearer $TOKEN"
```

## 🐛 Troubleshooting

### Common Issues

**1. Cannot connect to MongoDB**
```powershell
# Check if chat_db is running
docker ps | findstr chat

# View logs
docker-compose logs chat_db

# Restart
docker-compose restart chat_db
```

**2. Chat service won't start**
- Verify MongoDB is running on port 27019
- Check `application.properties` configuration
- Ensure Eureka server is running
- Check for port conflicts

**3. Messages not showing**
- F12 Developer Tools → Network tab
- Check API responses for errors
- Verify JWT token is valid
- Check conversation ID is correct

**4. File upload fails**
- Verify media-service is running
- Check file size (may have limits)
- Check file type is supported
- View media-service logs

**5. Conversation not created**
- Verify user-service is running
- Check friend relationship exists
- Verify JWT token has correct user ID
- Check chat-service logs

## 📊 Data Models

### Conversation
```typescript
{
  id: string;
  type: 'DIRECT' | 'GROUP';
  name?: string;  // For GROUP only
  participantIds: string[];
  lastMessageId?: string;
  lastMessageAt?: Date;
  createdBy: string;
  createdAt: Date;
  updatedAt: Date;
}
```

### Message
```typescript
{
  id: string;
  conversationId: string;
  senderId: string;
  senderName: string;
  senderAvatar?: string;
  type: 'TEXT' | 'IMAGE' | 'FILE' | 'SYSTEM';
  content: string;
  attachment?: {
    fileName: string;
    fileUrl: string;
    fileType: string;
    fileSize: number;
    thumbnailUrl?: string;
  };
  replyToMessageId?: string;
  reactions: Array<{
    userId: string;
    userName: string;
    emoji: string;
    createdAt: Date;
  }>;
  status: 'SENT' | 'DELIVERED' | 'READ';
  readByUserIds: string[];
  createdAt: Date;
  updatedAt: Date;
  editedAt?: Date;
  isEdited: boolean;
  isDeleted: boolean;
}
```

## 🚦 Status

### ✅ Completed
- [x] MongoDB setup in Docker
- [x] Chat service with REST APIs
- [x] Direct conversation creation
- [x] Send text messages
- [x] Send media (images, files)
- [x] Message history
- [x] Conversation list
- [x] Search conversations
- [x] Integration with user-service
- [x] Integration with media-service
- [x] Frontend chat UI (Messenger-style)
- [x] Chat sidebar component
- [x] Chat message area component
- [x] "Nhắn tin" button in Friends list

### 🔄 Ready to Implement
- [ ] WebSocket real-time updates
- [ ] Typing indicators
- [ ] Read receipts (tick marks)
- [ ] Online/offline status
- [ ] Message reactions (emoji)
- [ ] Edit messages
- [ ] Delete messages
- [ ] Reply to messages
- [ ] Forward messages
- [ ] Group chat
- [ ] Voice messages
- [ ] Video calls

## 📝 Notes

### Design Decisions
1. **Direct MongoDB Connection**: Chat-service build trong IDE, không cần Docker container
2. **Media Upload Flow**: Upload trước → Lấy URL → Gửi message (đơn giản, reliable)
3. **User Info Caching**: Lưu senderName và senderAvatar trong message (giảm calls)
4. **REST First**: Implement REST APIs trước, WebSocket sau (incremental development)

### Performance Considerations
- Message pagination: 50 messages per page
- Conversation list: 20 conversations per page
- Auto-scroll: Debounced for performance
- Image thumbnails: Generated by media-service

### Security
- All endpoints require JWT authentication
- User can only access their own conversations
- File upload has type and size restrictions
- XSS prevention in message content

## 🤝 Contributing

### Adding New Features
1. Update backend service (chat-service)
2. Add API endpoints if needed
3. Update frontend components
4. Test thoroughly
5. Update documentation

### Code Style
- Backend: Follow Spring Boot best practices
- Frontend: React functional components with hooks
- TypeScript for type safety
- Meaningful variable and function names

## 📞 Support

For issues or questions:
1. Check [CHAT-QUICK-START.md](./CHAT-QUICK-START.md)
2. Check [CHAT-FEATURE-IMPLEMENTATION.md](./CHAT-FEATURE-IMPLEMENTATION.md)
3. Review service logs
4. Open an issue in the repository

---

**Built with ❤️ for CTU Connect**
