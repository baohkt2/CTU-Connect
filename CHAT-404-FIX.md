# Chat Feature - Fix 404 Errors on New Conversations

## Issues Fixed

### 1. Frontend: Duplicate conversation creation attempts
**Problem**: Logic trong ChatSidebar chỉ tạo conversation khi `conversations.length === 0`, gây ra vấn đề khi đã có conversations khác nhưng chưa có conversation với friend này.

**Solution**: 
- Bỏ điều kiện `conversations.length === 0`
- Thêm state `creatingConversation` để prevent duplicate calls
- useEffect chỉ phụ thuộc vào `friendUserId`, không phụ thuộc vào `conversations`

**File**: `client-frontend/src/components/chat/ChatSidebar.tsx`

```typescript
// Before ❌
useEffect(() => {
  if (friendUserId && conversations.length === 0) {
    createOrGetConversationWithFriend(friendUserId);
  }
}, [friendUserId, conversations]);

// After ✅
const [creatingConversation, setCreatingConversation] = useState(false);

useEffect(() => {
  if (friendUserId && !creatingConversation) {
    createOrGetConversationWithFriend(friendUserId);
  }
}, [friendUserId]);

const createOrGetConversationWithFriend = async (friendId: string) => {
  if (creatingConversation) return; // Prevent duplicate calls
  
  try {
    setCreatingConversation(true);
    const response = await apiClient.post(`/chats/conversations/direct/${friendId}`);
    if (response && response.id) {
      onSelectConversation(response.id);
      await loadConversations();
    }
  } catch (error) {
    console.error('Error creating conversation:', error);
    toast.error('Không thể tạo cuộc trò chuyện');
  } finally {
    setCreatingConversation(false);
  }
};
```

### 2. Frontend: 404 error when loading messages from empty conversation
**Problem**: Khi conversation mới tạo chưa có messages, API trả về 404, frontend show error toast không cần thiết.

**Solution**: Catch 404 error và set empty messages array thay vì show error.

**File**: `client-frontend/src/components/chat/ChatMessageArea.tsx`

```typescript
// Before ❌
const loadMessages = async () => {
  try {
    const response = await apiClient.get(`/chats/messages/conversation/${conversationId}`);
    setMessages(response.content?.reverse() || []);
  } catch (error) {
    toast.error('Không thể tải tin nhắn'); // Shows error even for empty conversations
  }
};

// After ✅
const loadMessages = async () => {
  try {
    const response = await apiClient.get(`/chats/messages/conversation/${conversationId}`);
    setMessages(response.content?.reverse() || []);
  } catch (error) {
    console.error('Error loading messages:', error);
    // Don't show error for 404 (empty conversation)
    if (error.response?.status === 404) {
      setMessages([]); // Just set empty messages
    } else {
      toast.error('Không thể tải tin nhắn');
    }
  }
};
```

### 3. Frontend: Suppress error toast when conversations list is empty
**Problem**: Show error toast khi chưa có conversations nào (404 là expected cho user mới).

**Solution**: Chỉ show error toast khi status code không phải 404.

**File**: `client-frontend/src/components/chat/ChatSidebar.tsx`

```typescript
// After ✅
const loadConversations = async () => {
  try {
    const response = await apiClient.get('/chats/conversations');
    setConversations(response.content || []);
  } catch (error) {
    console.error('Error loading conversations:', error);
    // Don't show error toast if it's just empty conversations
    if (error.response?.status !== 404) {
      toast.error('Không thể tải danh sách trò chuyện');
    }
  }
};
```

### 4. Backend: Get real user info for participants
**Problem**: `getParticipantInfo` trong ConversationService trả về placeholder data (TODO comment).

**Solution**: Gọi UserService để lấy thông tin user thực tế.

**File**: `chat-service/src/main/java/com/ctuconnect/service/ConversationService.java`

```java
// Before ❌
private ParticipantInfo getParticipantInfo(String userId) {
    ParticipantInfo info = new ParticipantInfo();
    info.setUserId(userId);
    // TODO: Implement thực tế sau khi tích hợp với UserService
    info.setUserName("User " + userId);
    info.setUserAvatar("");
    return info;
}

// After ✅
private ParticipantInfo getParticipantInfo(String userId) {
    ParticipantInfo info = new ParticipantInfo();
    info.setUserId(userId);
    
    try {
        // Get user info from user-service
        Map<String, Object> userInfo = userService.getUserInfo(userId);
        info.setUserName((String) userInfo.getOrDefault("fullName", "User " + userId));
        info.setUserAvatar((String) userInfo.getOrDefault("avatarUrl", ""));
    } catch (Exception e) {
        log.warn("Failed to get user info for userId: {}, using defaults", userId, e);
        info.setUserName("User " + userId);
        info.setUserAvatar("");
    }
    
    return info;
}
```

## Flow After Fix

### Starting a new chat with a friend

```
User clicks "Nhắn tin" on friend
    ↓
Navigate to /messages?userId={friendId}
    ↓
ChatSidebar useEffect triggers
    ↓
Check if not already creating conversation
    ↓
POST /chats/conversations/direct/{friendId}
    ├─ Backend checks if conversation exists
    ├─ If exists: return existing
    └─ If not exists: create new and return
    ↓
Conversation ID received
    ↓
onSelectConversation(conversationId)
    ↓
ChatMessageArea loads messages
    ├─ If messages exist: display them
    └─ If 404 (empty): display empty state (no error toast)
    ↓
Ready to chat! ✅
```

## Expected Behavior

### For New Conversation
1. ✅ Click "Nhắn tin" creates conversation immediately
2. ✅ No error toast if conversation is empty
3. ✅ Shows empty chat window ready for first message
4. ✅ Participant names and avatars load from user-service

### For Existing Conversation
1. ✅ Returns existing conversation (no duplicate)
2. ✅ Loads message history
3. ✅ Shows conversation in sidebar with correct info

### Error Handling
1. ✅ 404 on empty conversations: Gracefully handled (no error toast)
2. ✅ 404 on empty message list: Gracefully handled (no error toast)
3. ✅ Network errors: Show appropriate error toast
4. ✅ User info fetch fails: Falls back to default "User {id}"

## Testing Checklist

### New Conversation
- [ ] Click "Nhắn tin" on friend who you never chatted with
- [ ] Verify conversation created immediately
- [ ] Verify no 404 error toasts shown
- [ ] Verify empty chat window displayed
- [ ] Verify friend's name and avatar shown correctly
- [ ] Send first message
- [ ] Verify message appears in chat
- [ ] Verify conversation appears in sidebar

### Existing Conversation
- [ ] Click "Nhắn tin" on friend you already chatted with
- [ ] Verify returns to existing conversation
- [ ] Verify message history loads
- [ ] Verify no duplicate conversation created

### Edge Cases
- [ ] Fast clicking "Nhắn tin" multiple times
  - Should not create duplicate conversations (creatingConversation guard)
- [ ] User-service is down
  - Should show fallback names "User {id}"
  - Should not crash or show repeated errors
- [ ] Network issues
  - Should show appropriate error messages
  - Should allow retry

## Files Modified

### Frontend
1. `client-frontend/src/components/chat/ChatSidebar.tsx`
   - Fixed conversation creation logic
   - Added creatingConversation state
   - Suppress 404 error for empty conversations list

2. `client-frontend/src/components/chat/ChatMessageArea.tsx`
   - Gracefully handle 404 when loading messages from empty conversation

### Backend
1. `chat-service/src/main/java/com/ctuconnect/service/ConversationService.java`
   - Integrated real user info from UserService
   - Added error handling for user info fetch
   - Added Map import

## Build Status
✅ **Backend**: Built successfully
✅ **Frontend**: TypeScript errors resolved

## Impact
- **User Experience**: Smoother chat creation, no confusing error messages
- **Reliability**: Better error handling, graceful fallbacks
- **Data Quality**: Real user names and avatars instead of placeholders

## Related Files
- Backend: `ConversationService.java`, `UserService.java`
- Frontend: `ChatSidebar.tsx`, `ChatMessageArea.tsx`
- Documentation: `CHAT-FEATURE-IMPLEMENTATION.md`, `CHAT-QUICK-START.md`
