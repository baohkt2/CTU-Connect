# Chat Authentication Client Fix

## Critical Issue
Khi click "Nhắn tin" từ Friends list, trang `/messages` load xong nhưng ngay lập tức redirect về `/login` mặc dù user đã đăng nhập.

## Root Cause

### Hệ thống có 2 API clients khác nhau:

**1. `api` (from `@/lib/api.ts`) - Đúng ✅**
- Sử dụng **HttpOnly cookies** cho authentication
- Auto refresh token khi 401
- Được dùng bởi: AuthContext, authService, userService
- Base URL: `http://localhost:8090/api`
- withCredentials: true

**2. `apiClient` (from `@/shared/config/api-client.ts`) - Sai ❌**
- Sử dụng **localStorage JWT token**
- Không có cookies
- KHÔNG tương thích với authentication system
- Chat components đang dùng client này!

### Vấn đề

```typescript
// Chat components đang dùng SAI client
import { apiClient } from '@/shared/config/api-client'; // ❌ NO COOKIES!

// Khi call API:
await apiClient.get('/chats/conversations'); 
// → Không gửi cookies → Backend trả về 401
// → apiClient redirect về '/login'
// → User bị đá ra mặc dù đã login!
```

## Solution

### Thay đổi Chat Components để dùng đúng API client

#### 1. ChatSidebar.tsx

```typescript
// Before ❌
import { apiClient } from '@/shared/config/api-client';

const response = await apiClient.get('/chats/conversations');
const data = response.content; // Wrong structure

// After ✅
import api from '@/lib/api';

const response = await api.get('/chats/conversations');
const data = response.data.content; // Correct structure
```

#### 2. ChatMessageArea.tsx

```typescript
// Before ❌
import { apiClient } from '@/shared/config/api-client';

const response = await apiClient.get('/chats/messages/...');
const messages = response.content;

const sendResponse = await apiClient.post('/chats/messages', data);
const newMessage = sendResponse;

// After ✅
import api from '@/lib/api';

const response = await api.get('/chats/messages/...');
const messages = response.data.content; // Note: response.data

const sendResponse = await api.post('/chats/messages', data);
const newMessage = sendResponse.data; // Note: response.data
```

## Key Differences

| Feature | `api` (✅ Correct) | `apiClient` (❌ Wrong) |
|---------|-------------------|----------------------|
| Auth Method | HttpOnly Cookies | localStorage JWT |
| withCredentials | true | false |
| Auto Refresh | Yes (401 → refresh) | No |
| Response Structure | `response.data.xxx` | `response.xxx` |
| Used By | Auth, User, Posts | Nothing (legacy) |
| Compatible with Backend | ✅ Yes | ❌ No |

## Why This Happened

1. **Two API client patterns**: Project có cả cookie-based và token-based
2. **Chat feature added later**: Dùng nhầm client pattern
3. **Different interceptors**: `apiClient` có interceptor khác với `api`

## Response Structure Changes

### api (Axios response)
```typescript
const response = await api.get('/endpoint');
// response.data = { content: [...], success: true, ... }
// Access data: response.data.content
```

### apiClient (Custom interceptor)
```typescript
const response = await apiClient.get('/endpoint');
// response = { content: [...], success: true, ... }
// Access data: response.content
```

**Fix**: All chat API calls now use `response.data.xxx` pattern.

## Files Modified

### 1. `client-frontend/src/components/chat/ChatSidebar.tsx`
**Changes**:
- Import: `apiClient` → `api`
- Response access: `response.content` → `response.data.content`
- Response access: `response.id` → `response.data.id`

### 2. `client-frontend/src/components/chat/ChatMessageArea.tsx`
**Changes**:
- Import: `apiClient` → `api`
- Response access: `response.content` → `response.data.content`
- Response access: `response.url` → `response.data.url`
- All API calls updated to use `response.data.*`

## Testing

### Verify Authentication Works

1. **Login normally**: Go to `/login`, enter credentials
2. **Check cookies**: 
   ```javascript
   // In browser console
   document.cookie // Should show cookies
   ```
3. **Navigate to /messages**: Should NOT redirect to login
4. **Click "Nhắn tin" from Friends**: Should work correctly

### Verify API Calls Work

1. **Open DevTools → Network tab**
2. **Navigate to /messages**
3. **Check requests**:
   - Should see `/chats/conversations` with status 200
   - Request headers should include cookies
   - No redirect to /login

### Verify Chat Functions

1. **Create conversation**: Click "Nhắn tin" from friend
2. **Load messages**: Should load without 401 errors
3. **Send message**: Type and send a message
4. **Upload file**: Try uploading an image

All should work without authentication errors.

## Why Cookies > localStorage for Auth

### Advantages of HttpOnly Cookies
1. **XSS Protection**: JavaScript cannot access HttpOnly cookies
2. **Auto-sent**: Browser automatically includes cookies in requests
3. **Secure**: Can set Secure flag for HTTPS only
4. **Refresh Logic**: Easier to implement auto-refresh

### Disadvantages of localStorage
1. **XSS Vulnerable**: JavaScript can steal tokens
2. **Manual Management**: Must manually add to headers
3. **No Auto-refresh**: Need custom logic
4. **CORS Issues**: More complex CORS configuration

## Backend Expectations

The backend (API Gateway + Services) expects:
```http
GET /api/chats/conversations
Cookie: accessToken=xxx; refreshToken=yyy; ...
```

NOT:
```http
GET /api/chats/conversations
Authorization: Bearer xxx
```

## Migration Path

### If You Need to Use apiClient Elsewhere

Don't! The project uses **cookie-based auth**. All new features should use:

```typescript
import api from '@/lib/api';

// All API calls
await api.get('/endpoint', { withCredentials: true });
await api.post('/endpoint', data, { withCredentials: true });
```

### Deprecating apiClient

Consider removing or refactoring `@/shared/config/api-client.ts` to avoid confusion.

## Related Issues Fixed

This fix resolves:
1. ✅ Redirect to login when accessing /messages
2. ✅ 401 errors on conversation creation
3. ✅ 401 errors on message loading
4. ✅ 401 errors on message sending
5. ✅ 401 errors on file upload

## Summary

**Problem**: Chat used wrong API client (localStorage JWT instead of cookies)

**Solution**: Changed to use `api` from `@/lib/api` with cookie authentication

**Result**: Chat now works correctly with the authentication system

---

**Status**: ✅ FIXED
**Priority**: CRITICAL
**Impact**: Chat feature now fully functional with authentication
