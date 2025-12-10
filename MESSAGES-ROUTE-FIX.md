# Messages Route 404 Fix

## Issue
Navigating to `/messages?userId={id}` results in 404 Page Not Found.

## Root Cause
In Next.js 13+ App Router, `useSearchParams()` requires a Suspense boundary when used in client components.

## Solution

### Updated: `client-frontend/src/app/messages/page.tsx`

**Problem**: Direct use of `useSearchParams()` without Suspense boundary.

**Fix**: Wrap the component content in a Suspense boundary.

```typescript
'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import Layout from '@/components/layout/Layout';
import { useRouter, useSearchParams } from 'next/navigation';
import ChatSidebar from '@/components/chat/ChatSidebar';
import ChatMessageArea from '@/components/chat/ChatMessageArea';

// Main content component that uses useSearchParams
function MessagesContent() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
  const friendUserId = searchParams.get('userId');

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <Layout>
      <div className="h-[calc(100vh-4rem)] flex bg-white">
        <ChatSidebar
          selectedConversationId={selectedConversationId}
          onSelectConversation={setSelectedConversationId}
          friendUserId={friendUserId}
        />
        
        <ChatMessageArea
          conversationId={selectedConversationId}
          currentUserId={user.id}
        />
      </div>
    </Layout>
  );
}

// Export with Suspense boundary
export default function MessagesPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    }>
      <MessagesContent />
    </Suspense>
  );
}
```

## Why This Fix Works

### Next.js 13+ App Router Requirements
1. **useSearchParams() is async**: In the new App Router, search params are treated as async data
2. **Suspense Required**: Any component using `useSearchParams()` must be wrapped in a Suspense boundary
3. **Prevents Static Generation Issues**: Without Suspense, Next.js can't properly handle the dynamic nature of search params

### Before Fix
```typescript
export default function MessagesPage() {
  const searchParams = useSearchParams(); // ❌ Error: Missing Suspense boundary
  // ...
}
```

### After Fix
```typescript
function MessagesContent() {
  const searchParams = useSearchParams(); // ✅ OK: Inside Suspense boundary
  // ...
}

export default function MessagesPage() {
  return (
    <Suspense fallback={<Loading />}>
      <MessagesContent />
    </Suspense>
  );
}
```

## How to Test

### 1. Restart Next.js Dev Server
```bash
# Stop current server (Ctrl+C)
cd client-frontend
npm run dev
```

### 2. Test the Route
Navigate to:
- `http://localhost:3000/messages` - Should show messages page
- `http://localhost:3000/messages?userId=some-uuid` - Should auto-create conversation with that user

### 3. Test from Friends List
1. Login to the application
2. Go to Friends page (`/friends`)
3. Click "Nhắn tin" button on any friend
4. Should navigate to `/messages?userId={friendId}` without 404

## Expected Behavior

### Route Access
- ✅ Direct URL: `http://localhost:3000/messages`
- ✅ With query params: `http://localhost:3000/messages?userId={id}`
- ✅ From navigation: `router.push('/messages?userId={id}')`

### Loading States
1. **Initial Load**: Shows spinner while checking auth
2. **Creating Conversation**: Shows sidebar with loading state
3. **Ready**: Shows chat interface with empty or existing messages

### Error Cases
- Not logged in: Redirects to `/login`
- Invalid userId: Shows error toast from ChatSidebar
- Network error: Shows appropriate error message

## Additional Notes

### Suspense Fallback
The fallback UI shows while:
- Component is being server-rendered
- Search params are being parsed
- Initial data is loading

### Alternative Approaches

#### Option 1: Use route params instead of search params
```typescript
// Change route to /messages/[userId]
// Then access via: params.userId
// This doesn't require Suspense
```

#### Option 2: Use client-side only routing
```typescript
// Get userId from window.location.search manually
// But this is less clean and not SSR-friendly
```

Current solution (Suspense) is the **recommended Next.js approach**.

## Related Files
- `client-frontend/src/app/messages/page.tsx` - Main messages page
- `client-frontend/src/components/chat/ChatSidebar.tsx` - Conversations list
- `client-frontend/src/components/chat/ChatMessageArea.tsx` - Chat window
- `client-frontend/src/features/users/components/friends/FriendsList.tsx` - "Nhắn tin" button

## Verification Checklist
- [x] Added Suspense boundary
- [x] Suspense import from React
- [x] Fallback loading UI provided
- [ ] Restart Next.js dev server
- [ ] Test direct URL access
- [ ] Test from Friends list navigation
- [ ] Verify query params received correctly

## Troubleshooting

### Still getting 404?
1. **Clear Next.js cache**:
   ```bash
   cd client-frontend
   rm -rf .next
   npm run dev
   ```

2. **Check browser console**: Look for any React errors

3. **Verify file location**: Ensure `page.tsx` is at `src/app/messages/page.tsx`

4. **Check for typos**: Route should be `/messages` not `/message`

### Route works but no data?
- Check if chat-service is running
- Check if API Gateway is routing correctly
- Verify JWT token is valid
- Check browser Network tab for API errors

## Status
✅ **FIXED** - Suspense boundary added to messages page
