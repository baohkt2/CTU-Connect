# Auth Login Redirect Fix

## Issue
When accessing protected routes (like `/messages`) without authentication, the app redirects to `/auth/login` which shows a 404 page because that route doesn't exist.

## Root Cause
The `api-client.ts` file has incorrect redirect path in `handleUnauthorized()` method.

**Actual login route**: `/login`  
**Incorrect redirect**: `/auth/login` ❌

## Solution

### File: `client-frontend/src/shared/config/api-client.ts`

```typescript
// Before ❌
private handleUnauthorized() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('auth_token');
    window.location.href = '/auth/login'; // Wrong path!
  }
}

// After ✅
private handleUnauthorized() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('auth_token');
    window.location.href = '/login'; // Correct path!
  }
}
```

## App Route Structure

```
client-frontend/src/app/
├── login/           ✅ Exists at /login
│   └── page.tsx
├── register/        ✅ Exists at /register
│   └── page.tsx
├── messages/        ✅ Exists at /messages
│   └── page.tsx
└── friends/         ✅ Exists at /friends
    └── page.tsx

Note: There is NO /auth/login route
```

## How This Affects Users

### Before Fix
1. User not logged in tries to access `/messages`
2. API call fails with 401 Unauthorized
3. `api-client.ts` catches 401
4. Redirects to `/auth/login`
5. **404 Page** shows (route doesn't exist)
6. User is confused 😕

### After Fix
1. User not logged in tries to access `/messages`
2. API call fails with 401 Unauthorized
3. `api-client.ts` catches 401
4. Redirects to `/login`
5. **Login page** shows correctly ✅
6. User can log in 😊

## Related Code Locations

### Correct Redirects (Already Fixed)
1. **messages/page.tsx**: `router.push('/login')` ✅
2. **Other protected pages**: Use `/login` ✅

### Fixed
1. **api-client.ts**: Changed from `/auth/login` to `/login` ✅

## Testing

### Test Unauthorized Access
1. **Clear token**: 
   ```javascript
   // In browser console
   localStorage.removeItem('auth_token');
   ```

2. **Try to access protected route**:
   ```
   http://localhost:3000/messages
   ```

3. **Expected behavior**:
   - Should redirect to `http://localhost:3000/login`
   - Should show login form (not 404)

### Test API 401 Response
1. **Login first**
2. **Manipulate token to make it invalid**:
   ```javascript
   localStorage.setItem('auth_token', 'invalid_token');
   ```

3. **Try to make API call** (e.g., load conversations)

4. **Expected behavior**:
   - API returns 401
   - Auto-redirect to `/login`
   - Token cleared from localStorage
   - Shows login page (not 404)

### Test Normal Login Flow
1. Go to `http://localhost:3000/login`
2. Enter credentials
3. Click Login
4. Should redirect to home or previous page

## Additional Notes

### Why `/login` not `/auth/login`?

In Next.js 13+ App Router, routes are determined by folder structure:
- `src/app/login/page.tsx` → `/login`
- `src/app/auth/login/page.tsx` → `/auth/login`

The project uses the first pattern, so redirect must be `/login`.

### Where 401 Can Occur

The `handleUnauthorized()` method is triggered when:
1. JWT token expired
2. JWT token invalid
3. JWT token missing but endpoint requires auth
4. Backend returns 401 for any reason

Common scenarios:
- Loading conversations: `GET /chats/conversations`
- Creating conversation: `POST /chats/conversations/direct/{id}`
- Sending message: `POST /chats/messages`
- Any protected API endpoint

### Best Practice

Always use consistent redirect paths across the application:
```typescript
// ✅ Good - Use route constant
const LOGIN_ROUTE = '/login';
window.location.href = LOGIN_ROUTE;

// ❌ Bad - Hardcoded different values
window.location.href = '/auth/login'; // One place
router.push('/login'); // Another place
```

## Files Modified
1. `client-frontend/src/shared/config/api-client.ts` - Fixed redirect path

## Verification Checklist
- [x] Fixed redirect path in api-client.ts
- [ ] Clear browser localStorage
- [ ] Test accessing /messages without login
- [ ] Verify redirects to /login (not /auth/login)
- [ ] Verify login page shows (not 404)
- [ ] Test login flow works after redirect

## Impact
- **User Experience**: Users see login page instead of 404 when unauthorized
- **Developer Experience**: Consistent redirect behavior across app
- **Security**: Proper handling of unauthorized access

## Status
✅ **FIXED** - Redirect path corrected to `/login`

---

**Related Documentation**:
- CHAT-ALL-FIXES-SUMMARY.md
- MESSAGES-ROUTE-FIX.md
