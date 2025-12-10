# Chat Feature - Import Path Fixes

## Issue
TypeScript error trong ChatSidebar.tsx và ChatMessageArea.tsx:
```
Cannot find module '@/lib/api-client' or its corresponding type declarations.
```

## Root Cause
Import path không đúng. File `api-client.ts` nằm ở `src/shared/config/api-client.ts` chứ không phải `src/lib/api-client.ts`.

## Solution

### Files Updated

#### 1. ChatSidebar.tsx
```typescript
// Before
import { apiClient } from '@/lib/api-client';

// After
import { apiClient } from '@/shared/config/api-client';
```

#### 2. ChatMessageArea.tsx
```typescript
// Before
import { apiClient } from '@/lib/api-client';

// After
import { apiClient } from '@/shared/config/api-client';
```

## Verification

### ApiClient Structure
File location: `client-frontend/src/shared/config/api-client.ts`

Exports:
```typescript
export class ApiClient {
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T>
  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T>
  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T>
  async patch<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T>
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T>
}

export const apiClient = new ApiClient(); // Singleton instance
```

### Usage in Components
Both components now correctly import and use:
```typescript
import { apiClient } from '@/shared/config/api-client';

// In ChatSidebar
const response = await apiClient.get('/api/chats/conversations');
const response = await apiClient.post(`/api/chats/conversations/direct/${friendId}`);

// In ChatMessageArea
const response = await apiClient.get(`/api/chats/messages/conversation/${conversationId}`);
const response = await apiClient.post('/api/chats/messages', messageData);
const response = await apiClient.post('/api/media/upload', formData);
```

## Status
✅ Fixed - TypeScript errors resolved

## Notes
- ApiClient automatically handles JWT token injection via interceptors
- Response interceptor returns `response.data` directly
- Handles 401 unauthorized by redirecting to login
- Base URL configured via `API_ENDPOINTS.BASE_URL`
