# Chat Real-time và Cải thiện UI - 10/12/2025

## Tổng quan
Tài liệu này ghi lại các cải thiện cho tính năng chat bao gồm WebSocket real-time, upload media, và cải thiện UI.

## 1. Các vấn đề đã fix

### 1.1. Upload Media không hiển thị tin nhắn
**Vấn đề:**
- Upload file/image lên media-service thành công
- Nhưng tin nhắn không được lưu và hiển thị

**Giải pháp:**
- Thêm trường `type` vào `SendMessageRequest.java`
- Frontend gửi `type: 'TEXT' | 'IMAGE' | 'FILE'` khi gửi tin nhắn
- Backend tự động xác định type dựa vào `fileType` của attachment

**File thay đổi:**
```
chat-service/src/main/java/com/ctuconnect/dto/request/SendMessageRequest.java
- Thêm: private Message.MessageType type;

client-frontend/src/components/chat/ChatMessageArea.tsx
- Cải thiện handleFileSelect() để xác định messageType
- Gửi type: 'IMAGE' hoặc 'FILE' cùng với attachment
```

### 1.2. Chat Real-time với WebSocket
**Vấn đề:**
- Tin nhắn không cập nhật real-time
- Phải reload để thấy tin nhắn mới

**Giải pháp:**
- Tích hợp WebSocket client với SockJS và STOMP
- Subscribe vào topic `/topic/conversation/{conversationId}`
- Nhận tin nhắn mới qua WebSocket và cập nhật UI ngay lập tức

**Dependencies đã có:**
```json
"@stomp/stompjs": "^7.0.0",
"sockjs-client": "^1.6.1",
"@types/sockjs-client": "^1.5.4"
```

**Implementation:**
```typescript
// client-frontend/src/components/chat/ChatMessageArea.tsx

useEffect(() => {
  if (!conversationId || !user) return;

  const connectWebSocket = () => {
    const socket = new SockJS('http://localhost:8090/api/ws/chat');
    const client = new Client({
      webSocketFactory: () => socket as any,
      connectHeaders: {
        'X-User-Id': user.id,
        'X-Username': user.email || user.fullName,
      },
      reconnectDelay: 5000,
      heartbeatIncoming: 4000,
      heartbeatOutgoing: 4000,
    });

    client.onConnect = () => {
      console.log('WebSocket connected');
      setIsConnected(true);

      // Subscribe to conversation messages
      client.subscribe(`/topic/conversation/${conversationId}`, (message: IMessage) => {
        const newMessage = JSON.parse(message.body);
        setMessages((prev) => {
          if (prev.some(m => m.id === newMessage.id)) {
            return prev; // Tránh duplicate
          }
          return [...prev, newMessage];
        });
      });
    };

    client.activate();
    stompClientRef.current = client;
  };

  connectWebSocket();

  return () => {
    if (stompClientRef.current) {
      stompClientRef.current.deactivate();
    }
  };
}, [conversationId, user]);
```

### 1.3. Cải thiện UI Chat

#### ChatSidebar
**Cải tiến:**
- Gradient header với màu từ blue-50 đến indigo-50
- Avatar với gradient background
- Badge hiển thị số tin nhắn chưa đọc (max 99+)
- Loading indicator khi đang tạo conversation
- Online status indicator cho direct chats
- Selected conversation có border-left màu blue

**File:** `client-frontend/src/components/chat/ChatSidebar.tsx`

#### ChatMessageArea
**Cải tiến:**
- **Chat Header mới:**
  - Hiển thị avatar và tên của conversation
  - Online status indicator
  - Action buttons: Call, Video, Info
  
- **Empty state đẹp hơn:**
  - Gradient background
  - Icon lớn với shadow
  - Text mô tả rõ ràng

- **Message bubbles:**
  - Màu xanh cho tin nhắn của mình
  - Màu xám cho tin nhắn từ người khác
  - Avatar hiển thị cho tin nhắn từ người khác
  - Timestamp và "Đã chỉnh sửa" indicator

**File:** `client-frontend/src/components/chat/ChatMessageArea.tsx`

#### Messages Page
**Cải tiến:**
- Load thông tin conversation details khi chọn
- Truyền thông tin conversation (name, avatar, isOnline) xuống ChatMessageArea
- Layout với rounded corners và shadow

**File:** `client-frontend/src/app/messages/page.tsx`

## 2. Navigation đến /messages

Navigation đã có sẵn trong Layout.tsx:
```typescript
{
  name: 'Tin nhắn',
  href: '/messages',
  icon: ChatBubbleLeftRightIcon,
  iconSolid: ChatIconSolid,
  badge: unreadCount  // Hiển thị số tin nhắn chưa đọc
}
```

Icon hiển thị cả trên:
- Desktop navigation bar (trên cùng)
- Mobile bottom navigation bar

## 3. Luồng gửi tin nhắn

### Text message:
1. User nhập text và nhấn Enter hoặc click Send
2. Frontend gửi POST `/chats/messages` với `{ conversationId, content, type: 'TEXT' }`
3. Backend lưu message và gửi qua WebSocket
4. Tất cả clients đang subscribe nhận được tin nhắn mới
5. UI update real-time

### Media message (Image/File):
1. User chọn file
2. Frontend upload lên POST `/media/upload`
3. Nhận về `cloudinaryUrl` và metadata
4. Frontend xác định type: 'IMAGE' nếu image/*, 'FILE' nếu khác
5. Gửi POST `/chats/messages` với `{ conversationId, content: fileName, type, attachment: {...} }`
6. Backend lưu message và gửi qua WebSocket
7. UI hiển thị image preview hoặc file download link

## 4. Cấu trúc Message

### Frontend (TypeScript):
```typescript
interface Message {
  id: string;
  content: string;
  senderId: string;
  senderName: string;
  senderAvatar?: string;
  type: 'TEXT' | 'IMAGE' | 'FILE';
  createdAt: string;
  isEdited: boolean;
  attachment?: {
    fileName: string;
    fileUrl: string;
    fileType: string;
    fileSize: number;
    thumbnailUrl?: string;
  };
}
```

### Backend (Java):
```java
public class Message {
    private String id;
    private String conversationId;
    private String senderId;
    private String senderName;
    private String senderAvatar;
    private MessageType type; // TEXT, IMAGE, FILE, SYSTEM
    private String content;
    private MessageAttachment attachment;
    private MessageStatus status; // SENT, DELIVERED, READ
    private LocalDateTime createdAt;
    private boolean isEdited;
}
```

## 5. WebSocket Configuration

### Backend (Spring Boot):
```java
// WebSocketConfig.java
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {
    
    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        config.enableSimpleBroker("/topic", "/queue");
        config.setApplicationDestinationPrefixes("/app");
        config.setUserDestinationPrefix("/user");
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws/chat")
                .setAllowedOriginPatterns("http://localhost:3000")
                .withSockJS();
    }
}
```

### Topics:
- `/topic/conversation/{conversationId}` - Tin nhắn mới
- `/topic/conversation/{conversationId}/typing` - Typing indicator

## 6. Testing

### Test chat cơ bản:
1. Login với 2 tài khoản khác nhau (2 browser)
2. Từ tài khoản A: Vào Friends → Click "Nhắn tin" với tài khoản B
3. Gửi tin nhắn text từ A → B nhận được ngay lập tức
4. Gửi tin nhắn từ B → A nhận được ngay lập tức

### Test upload media:
1. Click icon attachment (📎)
2. Chọn file image hoặc document
3. File được upload và hiển thị trong chat
4. Image hiển thị preview, file hiển thị link download

### Test UI:
1. Sidebar hiển thị danh sách conversations
2. Selected conversation có highlight
3. Chat header hiển thị đúng thông tin
4. Messages hiển thị đúng format (text, image, file)
5. Scroll mượt mà, messages mới xuất hiện ở cuối

## 7. Các file đã thay đổi

### Backend:
```
chat-service/src/main/java/com/ctuconnect/
├── dto/request/SendMessageRequest.java (Thêm type field)
└── service/MessageService.java (Xử lý type từ request)
```

### Frontend:
```
client-frontend/src/
├── app/messages/page.tsx (Load conversation details, truyền props)
├── components/chat/
│   ├── ChatMessageArea.tsx (WebSocket, upload media, chat header, UI)
│   └── ChatSidebar.tsx (Cải thiện UI)
```

## 8. Known Issues và Future Improvements

### Known Issues:
- ~~Duplicate conversations~~ (đã fix với synchronized method)
- ~~CORS duplicate headers~~ (đã fix)
- ~~Upload media không lưu message~~ (đã fix)

### Future Improvements:
1. **Typing indicator** - Hiển thị khi người khác đang gõ
2. **Read receipts** - Tích xanh khi tin nhắn đã đọc
3. **Message reactions** - Thả tim, like, emoji
4. **Reply to message** - Trả lời tin nhắn cụ thể
5. **Delete/Edit message** - Xóa hoặc chỉnh sửa tin nhắn đã gửi
6. **Voice messages** - Ghi âm và gửi tin nhắn thoại
7. **Group chat** - Tạo và quản lý nhóm chat
8. **Search messages** - Tìm kiếm tin nhắn trong conversation
9. **File preview** - Xem trước file PDF, video trong chat
10. **Push notifications** - Thông báo khi có tin nhắn mới (khi offline)

## 9. Environment Variables

Không cần thêm environment variables mới.

## 10. Dependencies

Tất cả dependencies đã có trong package.json:
```json
{
  "@stomp/stompjs": "^7.0.0",
  "sockjs-client": "^1.6.1",
  "@types/sockjs-client": "^1.5.4"
}
```

## 11. Conclusion

Tính năng chat đã hoàn thiện với:
✅ Real-time messaging qua WebSocket
✅ Upload và gửi media (image, file)
✅ UI/UX đẹp và mượt mà
✅ Navigation từ friends list
✅ Hiển thị online status
✅ Message history
✅ Responsive design

Hệ thống sẵn sàng cho việc mở rộng với các tính năng nâng cao như typing indicator, reactions, group chat, v.v.
