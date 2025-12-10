# Chat Feature - API Path Fix

## Issue
API calls đang bị duplicate prefix `/api/api/...` thay vì `/api/...`:
```
❌ http://localhost:8090/api/api/chats/conversations
❌ http://localhost:8090/api/api/chats/conversations/direct/{id}
❌ http://localhost:8090/api/api/chats/messages
❌ http://localhost:8090/api/api/media/upload
```

Dẫn đến lỗi **404 Not Found**.

## Root Cause

### ApiClient Configuration
File: `client-frontend/.env`
```env
NEXT_PUBLIC_API_URL=http://localhost:8090/api
```

ApiClient đã có `baseURL = http://localhost:8090/api`, nên khi gọi API **không nên** thêm `/api` prefix nữa.

### Incorrect Usage (Before)
```typescript
// ❌ Wrong - Results in /api/api/chats/...
await apiClient.get('/api/chats/conversations');
await apiClient.post('/api/chats/conversations/direct/${friendId}');
await apiClient.post('/api/chats/messages', data);
await apiClient.post('/api/media/upload', formData);
```

## Solution

### Files Updated

#### 1. ChatSidebar.tsx
```typescript
// ✅ Correct
const loadConversations = async () => {
  const response = await apiClient.get('/chats/conversations');
  // Results in: http://localhost:8090/api/chats/conversations
};

const createOrGetConversationWithFriend = async (friendId: string) => {
  const response = await apiClient.post(`/chats/conversations/direct/${friendId}`);
  // Results in: http://localhost:8090/api/chats/conversations/direct/{id}
};
```

#### 2. ChatMessageArea.tsx
```typescript
// ✅ Correct
const loadMessages = async () => {
  const response = await apiClient.get(`/chats/messages/conversation/${conversationId}`);
  // Results in: http://localhost:8090/api/chats/messages/conversation/{id}
};

const handleSendMessage = async () => {
  const response = await apiClient.post('/chats/messages', data);
  // Results in: http://localhost:8090/api/chats/messages
};

const handleFileSelect = async () => {
  // Upload file
  const uploadResponse = await apiClient.post('/media/upload', formData);
  // Results in: http://localhost:8090/api/media/upload
  
  // Send message with attachment
  const messageResponse = await apiClient.post('/chats/messages', messageData);
  // Results in: http://localhost:8090/api/chats/messages
};
```

## API Path Pattern

### Rule
When using `apiClient`, paths should **NOT** include `/api` prefix since it's already in `baseURL`.

```typescript
// ✅ Correct Pattern
baseURL = 'http://localhost:8090/api'
endpoint = '/chats/conversations'
result = 'http://localhost:8090/api/chats/conversations'

// ❌ Wrong Pattern
baseURL = 'http://localhost:8090/api'
endpoint = '/api/chats/conversations'
result = 'http://localhost:8090/api/api/chats/conversations' // DUPLICATE!
```

## Updated Endpoints

| Service | Old (Wrong) | New (Correct) | Final URL |
|---------|-------------|---------------|-----------|
| Load Conversations | `/api/chats/conversations` | `/chats/conversations` | `http://localhost:8090/api/chats/conversations` |
| Create Conversation | `/api/chats/conversations/direct/{id}` | `/chats/conversations/direct/{id}` | `http://localhost:8090/api/chats/conversations/direct/{id}` |
| Load Messages | `/api/chats/messages/conversation/{id}` | `/chats/messages/conversation/{id}` | `http://localhost:8090/api/chats/messages/conversation/{id}` |
| Send Message | `/api/chats/messages` | `/chats/messages` | `http://localhost:8090/api/chats/messages` |
| Upload Media | `/api/media/upload` | `/media/upload` | `http://localhost:8090/api/media/upload` |

## Verification

### Before Fix
```
Console Error:
:8090/api/api/chats/conversations:1 Failed to load resource: 404 (Not Found)
:8090/api/api/chats/conversations/direct/...:1 Failed to load resource: 404 (Not Found)
```

### After Fix
```
Expected Requests:
✅ GET  http://localhost:8090/api/chats/conversations
✅ POST http://localhost:8090/api/chats/conversations/direct/{friendId}
✅ GET  http://localhost:8090/api/chats/messages/conversation/{conversationId}
✅ POST http://localhost:8090/api/chats/messages
✅ POST http://localhost:8090/api/media/upload
```

## Testing

### 1. Load Conversations
```bash
# Should work now
curl http://localhost:8090/api/chats/conversations \
  -H "Authorization: Bearer {token}"
```

### 2. Create Conversation
```bash
# Should work now
curl -X POST http://localhost:8090/api/chats/conversations/direct/user123 \
  -H "Authorization: Bearer {token}"
```

### 3. Send Message
```bash
# Should work now
curl -X POST http://localhost:8090/api/chats/messages \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{"conversationId":"conv123","content":"Hello"}'
```

## Best Practices

### When Using ApiClient
1. **Never** add `/api` prefix to your endpoints
2. ApiClient baseURL already includes it
3. Use relative paths starting with service name: `/chats/...`, `/media/...`, `/users/...`

### Consistency
All API calls in the project should follow this pattern:
```typescript
// ✅ Good
apiClient.get('/users/me')
apiClient.post('/posts', data)
apiClient.get('/chats/conversations')

// ❌ Bad
apiClient.get('/api/users/me')
apiClient.post('/api/posts', data)
apiClient.get('/api/chats/conversations')
```

## Status
✅ **FIXED** - All chat API paths corrected

## Files Modified
1. `client-frontend/src/components/chat/ChatSidebar.tsx`
   - Fixed `loadConversations()` endpoint
   - Fixed `createOrGetConversationWithFriend()` endpoint

2. `client-frontend/src/components/chat/ChatMessageArea.tsx`
   - Fixed `loadMessages()` endpoint
   - Fixed `handleSendMessage()` endpoint
   - Fixed `handleFileSelect()` endpoints (both media upload and message send)

## Related Files
- `client-frontend/.env` - Contains `NEXT_PUBLIC_API_URL=http://localhost:8090/api`
- `client-frontend/src/shared/config/api-client.ts` - ApiClient implementation
- `client-frontend/src/shared/constants/api-endpoints.ts` - API endpoint definitions
