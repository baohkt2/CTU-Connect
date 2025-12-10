# Chat Sidebar Fixes - Dec 10, 2025

## Vấn đề đã sửa

### 1. **Hiển thị thông tin người gửi thay vì người nhận**
**Nguyên nhân**: 
- Frontend chỉ lấy `participants[0]` mà không kiểm tra đó có phải người nhận không
- Backend trả về tất cả participants (bao gồm cả user hiện tại)
- Cần lọc ra người nhận (không phải currentUserId)

**Giải pháp**:
- Fetch current user ID khi component mount
- Sử dụng `.find()` để tìm participant KHÔNG phải current user:
  ```typescript
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);

  useEffect(() => {
    const fetchCurrentUser = async () => {
      const response = await api.get('/users/me');
      setCurrentUserId(response.data.id);
    };
    fetchCurrentUser();
  }, []);

  const getConversationName = useCallback((conversation: Conversation) => {
    if (conversation.type === 'GROUP') {
      return conversation.name || 'Nhóm chat';
    }
    // Find the OTHER participant (not current user)
    const otherParticipant = conversation.participants.find(
      p => p.userId !== currentUserId
    );
    return otherParticipant?.userName || 'Người dùng';
  }, [currentUserId]);
  ```

**Key changes**:
- Thêm state `currentUserId`
- Callbacks depend on `currentUserId`
- Chỉ render sidebar khi `currentUserId` đã load: `{!isCollapsed && currentUserId && ...}`
- Log để debug: `console.log('Other participant:', otherParticipant)`

### 2. **Nút mở không hiển thị khi đóng sidebar**
**Nguyên nhân**: Nút toggle nằm trong div bị collapse (width: 0) nên không hiển thị được

**Giải pháp**:
- Tách nút toggle ra khỏi div collapse
- Wrap toàn bộ trong `<div className="relative">`
- Nút toggle LUÔN visible với position `absolute`:
  ```typescript
  <div className="relative">
    {/* Toggle button - ALWAYS visible */}
    <button
      onClick={() => setIsCollapsed(!isCollapsed)}
      className={`hidden sm:flex absolute ${isCollapsed ? 'left-4' : '-right-3'} top-4 z-30 w-8 h-8 ...`}
    >
      {isCollapsed ? <Menu /> : <X />}
    </button>

    {/* Collapsible sidebar content */}
    <div className={`${isCollapsed ? 'w-0' : 'w-full sm:w-80 md:w-96'} ...`}>
      ...
    </div>
  </div>
  ```

**Key points**:
- Position thay đổi: `left-4` khi collapsed, `-right-3` khi expanded
- `z-30` để nút luôn ở trên cùng
- Size tăng lên `w-8 h-8` (từ w-6 h-6) để dễ click hơn
- Border tăng lên `border-2` để nổi bật hơn

## Files Changed

1. **client-frontend/src/components/chat/ChatSidebar.tsx**
   - Fixed Participant interface
   - Fixed getConversationName and getConversationAvatar
   - Restructured toggle button layout
   - Added debug log for conversations

## Testing

### Test hiển thị tên thật:
1. Mở trang Messages
2. Chọn một conversation
3. Kiểm tra console log: `Loaded conversations: [...]`
4. Verify participants có `userName` và `userAvatar`
5. UI hiển thị tên thật của user

### Test toggle sidebar:
1. Desktop: Click nút toggle (dấu X)
2. Sidebar collapse về width 0
3. Nút toggle di chuyển sang `left-4` và đổi icon thành Menu
4. Click nút Menu
5. Sidebar expand lại
6. Nút toggle về position `-right-3` và đổi icon thành X

### Mobile:
- Toggle button ẩn (hidden sm:flex)
- Sidebar full width by default

## Backend Information

**ConversationService.java** (lines 288-304):
```java
private ParticipantInfo getParticipantInfo(String userId) {
    ParticipantInfo info = new ParticipantInfo();
    info.setUserId(userId);
    
    try {
        Map<String, Object> userInfo = userService.getUserInfo(userId);
        info.setUserName((String) userInfo.getOrDefault("fullName", "User " + userId));
        info.setUserAvatar((String) userInfo.getOrDefault("avatarUrl", ""));
    } catch (Exception e) {
        log.warn("Failed to get user info for userId: {}", userId, e);
        info.setUserName("User " + userId);
        info.setUserAvatar("");
    }
    
    return info;
}
```

**ParticipantInfo.java**:
```java
@Data
public class ParticipantInfo {
    private String userId;
    private String userName;
    private String userAvatar;
    private UserPresence.PresenceStatus presenceStatus;
    private LocalDateTime lastSeenAt;
    private boolean isAdmin;
}
```

## Notes

- Backend đang fetch user info từ user-service và map vào ParticipantInfo
- Nếu user-service không available, sẽ dùng fallback "User {userId}"
- Frontend giờ đây map đúng với field names từ backend
- Toggle button với animation smooth và position responsive
