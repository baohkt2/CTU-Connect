# Chat Feature - Testing Guide

## Quick Start

### 1. Khởi động services

```powershell
# Backend services (chạy từ IDE)
- chat-service (port 8086)
- user-service (port 8081) 
- media-service (port 8084)
- api-gateway (port 8090)

# Database (Docker)
docker-compose up -d chat_db

# Frontend
cd client-frontend
npm run dev
```

### 2. Test Chat Real-time

#### Chuẩn bị:
1. Mở 2 browser khác nhau (Chrome và Firefox) hoặc 2 tab incognito
2. Login với 2 tài khoản khác nhau

#### Test steps:

**Bước 1: Tạo conversation từ Friends list**
```
Browser A (User A):
1. Vào /friends
2. Tìm User B trong danh sách bạn bè
3. Click nút "Nhắn tin" 
4. → Redirect đến /messages?userId={userB_id}
5. → Tự động tạo conversation (nếu chưa có)
6. → Hiển thị chat window
```

**Bước 2: Gửi tin nhắn text**
```
Browser A:
1. Gõ "Xin chào" vào input box
2. Nhấn Enter hoặc click Send button
3. → Tin nhắn xuất hiện ngay lập tức
4. → Bubble màu xanh (bên phải)

Browser B:
1. Vào /messages
2. → Thấy conversation mới với User A
3. Click vào conversation
4. → Thấy tin nhắn "Xin chào" từ User A
5. → Bubble màu xám (bên trái)
6. → KHÔNG CẦN RELOAD PAGE
```

**Bước 3: Chat qua lại**
```
Browser B gửi: "Chào bạn!"
Browser A nhận được NGAY LẬP TỨC

Browser A gửi: "Bạn khỏe không?"
Browser B nhận được NGAY LẬP TỨC

✅ Real-time working!
```

**Bước 4: Upload image**
```
Browser A:
1. Click icon 📎 (attachment button)
2. Chọn 1 file ảnh (JPG, PNG)
3. → Loading indicator xuất hiện
4. → Upload hoàn tất
5. → Image hiển thị trong chat với preview
6. → User B nhận được ngay lập tức
7. Click vào image → Mở full size trong tab mới
```

**Bước 5: Upload file**
```
Browser A:
1. Click icon 📎
2. Chọn 1 file document (PDF, DOCX)
3. → Loading indicator xuất hiện
4. → Upload hoàn tất
5. → File hiển thị với icon và tên file
6. → User B nhận được ngay lập tức
7. Click vào file → Download/Open
```

### 3. Test UI/UX

#### ChatSidebar:
- ✅ Search conversations work
- ✅ Selected conversation có highlight (blue background, blue border-left)
- ✅ Avatar hiển thị đúng
- ✅ Online status (green dot)
- ✅ Last message preview
- ✅ Unread count badge (nếu có)
- ✅ Time ago format ("5 phút", "2 giờ", "3 ngày")

#### ChatMessageArea:
- ✅ Chat header hiển thị:
  - Avatar của conversation
  - Tên của conversation
  - Online status
  - Action buttons (Call, Video, Info)
- ✅ Messages:
  - Own messages: Blue bubble, bên phải
  - Other messages: Gray bubble, bên trái, có avatar
  - Timestamp hiển thị
  - "Đã chỉnh sửa" nếu edited
- ✅ Image preview
- ✅ File download link
- ✅ Auto scroll to bottom khi có message mới
- ✅ Input area với emoji-like placeholder "Aa"

#### Empty states:
- ✅ No conversations: Icon + text "Chưa có cuộc trò chuyện nào"
- ✅ No conversation selected: Icon + text "Chọn một cuộc trò chuyện để bắt đầu nhắn tin"

### 4. Test Edge Cases

#### Multiple messages nhanh:
```
Browser A gửi liên tục 5 tin nhắn:
"1"
"2"
"3"
"4"
"5"

Browser B phải nhận đúng 5 tin nhắn theo thứ tự
✅ Không bị duplicate
✅ Không bị mất tin nhắn
```

#### Connection lost:
```
1. Gửi tin nhắn khi có connection
2. Tắt internet
3. Gửi tin nhắn → Hiển thị error toast
4. Bật lại internet
5. WebSocket tự động reconnect (sau 5s)
6. Gửi tin nhắn → Work bình thường
```

#### Upload file lớn:
```
Upload file > 5MB
→ Loading lâu hơn
→ Progress indicator
→ Thành công và hiển thị
```

### 5. Browser Console Check

Mở Console (F12) và xem logs:

**Khi connect:**
```
STOMP: Connected to server
WebSocket connected
User {userId} connected with session {sessionId}
```

**Khi gửi tin nhắn:**
```
DEBUG: API Response successful: /chats/messages {...}
```

**Khi nhận tin nhắn:**
```
STOMP: <<< MESSAGE
Received message: {id: "...", content: "...", ...}
```

**Không được có:**
```
❌ CORS errors
❌ 404 errors
❌ WebSocket connection errors
❌ Duplicate messages logged
```

### 6. Network Tab Check

**WebSocket connection:**
```
Filter: WS
→ Thấy connection đến ws://localhost:8090/api/ws/chat
→ Status: 101 Switching Protocols
→ Frames tab: Thấy CONNECT, SUBSCRIBE, MESSAGE frames
```

**API calls:**
```
POST /api/chats/messages → 200 OK
GET /api/chats/conversations → 200 OK
GET /api/chats/messages/conversation/{id} → 200 OK
POST /api/media/upload → 200 OK
```

### 7. Database Check

```bash
# Connect to MongoDB
mongo mongodb://localhost:27017/chat_db

# Check conversations
db.conversations.find().pretty()
→ Xem type, participantIds, lastMessageAt

# Check messages
db.messages.find().sort({createdAt: -1}).limit(10).pretty()
→ Xem content, type, attachment, senderId
```

### 8. Performance Check

- ✅ Message gửi đi trong < 100ms
- ✅ WebSocket latency < 50ms
- ✅ Upload image < 2s (tùy kích thước và mạng)
- ✅ Load conversations < 500ms
- ✅ Load messages < 500ms
- ✅ UI smooth, không lag

### 9. Mobile Test (Optional)

```
1. Mở http://localhost:3000 trên điện thoại (cùng WiFi)
2. Login
3. Test chat giống như trên desktop
4. Kiểm tra responsive:
   - Bottom navigation bar
   - Touch gestures
   - Input keyboard
   - Image upload từ camera/gallery
```

### 10. Common Issues & Solutions

#### Issue 1: WebSocket không connect
```
Lỗi: WebSocket connection failed
Giải pháp:
- Check chat-service đang chạy (port 8086)
- Check api-gateway đang chạy (port 8090)
- Check CORS config trong SecurityConfig
```

#### Issue 2: Tin nhắn không real-time
```
Lỗi: Phải reload mới thấy tin nhắn mới
Giải pháp:
- Check console có log "WebSocket connected" không
- Check subscribe topic đúng không
- Restart chat-service
```

#### Issue 3: Upload file lỗi
```
Lỗi: Upload file không thành công
Giải pháp:
- Check media-service đang chạy (port 8084)
- Check Cloudinary credentials trong .env
- Check file size < 10MB
```

#### Issue 4: 404 khi tạo conversation
```
Lỗi: POST /api/chats/conversations/direct/{userId} → 404
Giải pháp:
- Check chat-service routes
- Check api-gateway routing
- Check userId có đúng không
```

#### Issue 5: Duplicate messages
```
Lỗi: Mỗi tin nhắn hiển thị 2 lần
Giải pháp:
- Check logic trong setMessages()
- Đảm bảo filter duplicate bằng message.id
- Check không gọi sendMessage 2 lần
```

## Success Criteria

✅ Gửi tin nhắn text real-time
✅ Gửi tin nhắn image với preview
✅ Gửi tin nhắn file với download link
✅ UI đẹp và responsive
✅ Navigation từ friends list
✅ Online status hiển thị
✅ No CORS errors
✅ No 404 errors
✅ No duplicate messages
✅ WebSocket stable connection

## Next Steps

Sau khi test thành công, có thể mở rộng với:
1. Typing indicator
2. Read receipts
3. Message reactions
4. Group chat
5. Voice messages
6. Video call
7. Search messages
8. Delete/Edit messages
