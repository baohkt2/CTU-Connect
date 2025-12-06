# User Service Endpoints Mapping

## Current Status & Required Changes

### Frontend Expectations vs Backend Reality

| Frontend Endpoint | Backend Status | Action Required |
|-------------------|----------------|-----------------|
| `GET /api/users/profile` | ❌ Missing (has `/me/profile`) | ✅ Add alias |
| `GET /api/users/:id` | ✅ Exists | ✅ OK |
| `PUT /api/users/profile` | ❌ Has `/users/:id` | ✅ Add alias for current user |
| `GET /api/users/search` | ✅ Exists | ✅ OK |
| `GET /api/users/:id/friends` | ✅ Exists | ✅ OK |
| `GET /api/users/friend-suggestions` | ✅ Exists | ✅ OK |
| `POST /api/users/:id/friend-request` | ❌ Wrong path | ✅ Adjust |
| `POST /api/users/:id/accept-friend` | ❌ Wrong path | ✅ Adjust |
| `GET /api/users/:id/mutual-friends-count` | ✅ Exists | ✅ OK |
| `GET /api/users/:id/timeline` | ❌ Missing | ✅ Create |
| `GET /api/users/:id/activities` | ⚠️ Check path | ✅ Verify |

## Implementation Plan

### 1. Profile Endpoints
- Add `GET /api/users/profile` as alias to `/me/profile`
- Add `PUT /api/users/profile` for updating current user

### 2. Friend Request Endpoints
- Keep existing complex paths for internal use
- Add simplified paths for frontend:
  - `POST /api/users/:targetUserId/friend-request` 
  - `POST /api/users/:requesterId/accept-friend`
  - `DELETE /api/users/:friendId/friend` (unfriend)

### 3. Timeline & Activities
- Add `GET /api/users/:id/timeline` - user's post timeline
- Verify `GET /api/users/:id/activities` - user activity feed

### 4. Additional Endpoints Needed
- `GET /api/users/:id/sent-requests` - sent friend requests
- `GET /api/users/:id/received-requests` - received friend requests
- `DELETE /api/users/:requestId/friend-request` - cancel friend request
