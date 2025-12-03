# API Mapping Documentation - Client Frontend ↔ Auth Service

## 🔗 Endpoints Mapping

### Authentication Endpoints
| Frontend Method | Backend Endpoint | Request Body | Response | Notes |
|----------------|-----------------|--------------|----------|-------|
| `authService.login()` | `POST /api/auth/login` | `LoginRequest` | `AuthResponse` | Tokens in cookies |
| `authService.register()` | `POST /api/auth/register` | `RegisterRequest` | `AuthResponse` | Tokens in cookies |
| `authService.logout()` | `POST /api/auth/logout` | `{refreshToken}` | `String message` | Clear cookies |
| `authService.refreshToken()` | `POST /api/auth/refresh-token` | `{refreshToken}` | `AuthResponse` | New tokens in cookies |
| `authService.forgotPassword()` | `POST /api/auth/forgot-password` | `{email}` | `String message` | - |
| `authService.resetPassword()` | `POST /api/auth/reset-password` | `{token, newPassword}` | `String message` | - |
| `authService.verifyEmail()` | `POST /api/auth/verify-email?token=xxx` | - | `String message` | Query param |
| `authService.getCurrentUser()` | `GET /api/auth/me` | - | `ApiResponse<User>` | Needs auth |

## 🍪 Cookie Management

### Backend Sets Cookies:
- `accessToken` - HttpOnly, Secure, 15 minutes
- `refreshToken` - HttpOnly, Secure, 7 days

### Frontend Cookie Access:
```typescript
// Helper methods trong authService
getAccessTokenFromCookie(): string | null
getRefreshTokenFromCookie(): string | null
isTokenExpired(): boolean
```

## 📊 Data Transfer Objects (DTOs)

### LoginRequest
```typescript
interface LoginRequest {
  email?: string;      // For CTU email login
  username?: string;   // For username login  
  password: string;
}
```

### RegisterRequest
```typescript
interface RegisterRequest {
  email: string;        // Must be @ctu.edu.vn
  password: string;     // Min 8 chars, complex
  username: string;     // 3-30 chars, alphanumeric + _
  fullName: string;     // 2-100 chars, letters only
  studentId: string;    // Format: CT123456, DT1234567
  faculty: string;      // From predefined list
  yearOfStudy: number;  // 1-6
}
```

### AuthResponse
```typescript
interface AuthResponse {
  user: User;
  accessToken: string | null;   // Null in response (in cookies)
  refreshToken: string | null;  // Null in response (in cookies)
}
```

### User
```typescript
interface User {
  id: string;
  email: string;
  username: string;
  fullName: string;
  avatar?: string;
  bio?: string;
  studentId?: string;
  faculty?: string;
  yearOfStudy?: number;
  isVerified: boolean;
  isOnline: boolean;
  createdAt: string;
  updatedAt: string;
}
```

## 🔧 Configuration Changes

### API Configuration (lib/api.ts)
- ✅ Added `withCredentials: true` for cookie support
- ✅ Auto-refresh token on 401 errors
- ✅ Base URL: `http://localhost:8080` (API Gateway)

### AuthService (services/authService.ts)
- ✅ All endpoints use `/api/auth/*` prefix
- ✅ Cookie-based authentication instead of manual token management
- ✅ Helper methods for cookie access
- ✅ Proper error handling with backend response format

### AuthContext (contexts/AuthContext.tsx)
- ✅ Removed manual cookie management with js-cookie
- ✅ Token validation using helper methods
- ✅ Auto-refresh logic on authentication check
- ✅ Proper logout with server call

### Environment Variables (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8080
NODE_ENV=development
```

## 🛡️ Security Features

### Implemented:
- ✅ HttpOnly cookies (không thể access từ JavaScript)
- ✅ Secure cookies (chỉ HTTPS trong production)
- ✅ Auto token refresh
- ✅ 401 error handling with retry
- ✅ Proper logout with server cleanup

### Validation Rules:
- ✅ CTU email only: `@ctu.edu.vn`
- ✅ Strong password: min 8 chars + uppercase + lowercase + digit + special char
- ✅ Username: 3-30 chars, starts with letter
- ✅ Student ID: format like CT123456, DT1234567

## 🎯 Frontend Form Validation

### LoginForm:
- Smart input detection (email vs username)
- Dynamic placeholder
- CTU email validation
- Strong password requirements

### RegisterForm:
- CTU email validation
- Student information fields
- Faculty dropdown (8 faculties)
- Year of study selection (1-6)
- Password strength indicator
- Real-time validation feedback

### ForgotPasswordForm:
- CTU email only
- Clear success/error messages

## 🔄 Authentication Flow

1. **Login/Register** → Backend sets cookies → Frontend receives user data
2. **Auto-check** → Read cookies → Validate token → Get current user
3. **API calls** → Cookies sent automatically → Auto-refresh on 401
4. **Logout** → Server call with refresh token → Clear cookies → Redirect

## ⚠️ Known Issues Fixed:

1. ❌ **Cookie mismatch**: Frontend used `auth_token`, Backend uses `accessToken`
   ✅ **Fixed**: Updated to use `accessToken` and `refreshToken`

2. ❌ **Response structure**: Frontend expected tokens in response body
   ✅ **Fixed**: Tokens now in cookies, only user data in response

3. ❌ **API endpoints**: Inconsistent endpoint paths
   ✅ **Fixed**: All use `/api/auth/*` pattern

4. ❌ **Manual token management**: Using js-cookie manually
   ✅ **Fixed**: Backend handles cookies, frontend uses helper methods

## 🚀 Ready for Integration

The client-frontend is now fully aligned with auth-service:
- ✅ Correct API endpoints
- ✅ Proper request/response handling  
- ✅ Secure cookie-based authentication
- ✅ Auto-refresh token mechanism
- ✅ CTU-specific validation rules
- ✅ Comprehensive error handling

All authentication flows should work seamlessly with the backend!
