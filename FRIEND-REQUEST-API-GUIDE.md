# Friend Request API Guide

## API Endpoints

### 1. Get RECEIVED Friend Requests ONLY
```
GET /api/users/me/friend-requests
```
**Purpose:** Get list of friend requests that OTHER users sent TO current user

**Response:**
```json
[
  {
    "id": "user-id",
    "fullName": "User Name",
    "email": "user@example.com",
    "requestType": "RECEIVED",  // Always "RECEIVED" for this endpoint
    "mutualFriendsCount": 5
  }
]
```

**UI Usage:** Display in "Friend Requests" tab - these are requests you can ACCEPT or REJECT

---

### 2. Get SENT Friend Requests ONLY
```
GET /api/users/me/friend-requested
```
**Purpose:** Get list of friend requests that current user sent TO other users

**Response:**
```json
[
  {
    "id": "user-id",
    "fullName": "User Name",
    "email": "user@example.com",
    "requestType": "SENT",  // Always "SENT" for this endpoint
    "mutualFriendsCount": 3
  }
]
```

**UI Usage:** Display separately - these are requests you sent and can CANCEL

---

### 3. Get ALL Friend Requests (Optional)
```
GET /api/users/me/friend-requests/all
```
**Purpose:** Get combined list of both sent and received requests

**Response:**
```json
[
  {
    "id": "user-1",
    "fullName": "John Doe",
    "requestType": "RECEIVED",  // From other user to you
    "mutualFriendsCount": 5
  },
  {
    "id": "user-2", 
    "fullName": "Jane Smith",
    "requestType": "SENT",  // From you to other user
    "mutualFriendsCount": 3
  }
]
```

**UI Usage:** If you want to show both in one tab, use requestType to distinguish

---

## UI Implementation Recommendations

### Option 1: Separate Tabs (Recommended)
```tsx
// Tab 1: "Friend Requests" - Show RECEIVED only
GET /api/users/me/friend-requests
// Show "Accept" and "Reject" buttons

// Tab 2: "Sent Requests" - Show SENT only  
GET /api/users/me/friend-requested
// Show "Cancel" button
```

### Option 2: Combined Tab with Sections
```tsx
// Get all requests
GET /api/users/me/friend-requests/all

// Group by requestType
const received = allRequests.filter(r => r.requestType === "RECEIVED");
const sent = allRequests.filter(r => r.requestType === "SENT");

// Show in sections
<Section title="Received Requests">
  {received.map(r => <RequestCard type="received" />)}
</Section>

<Section title="Sent Requests">
  {sent.map(r => <RequestCard type="sent" />)}
</Section>
```

---

## Request Flow Example

### User A sends request to User B

1. **User A clicks "Add Friend" on User B's profile**
   ```
   POST /api/users/me/invite/{userB-id}
   ```

2. **User A checks their sent requests**
   ```
   GET /api/users/me/friend-requested
   Response: [{ id: userB-id, requestType: "SENT" }]
   ```
   ✅ User B appears in User A's "Sent Requests"

3. **User B checks their received requests**
   ```
   GET /api/users/me/friend-requests
   Response: [{ id: userA-id, requestType: "RECEIVED" }]
   ```
   ✅ User A appears in User B's "Friend Requests"

4. **User B accepts the request**
   ```
   POST /api/users/me/accept-invite/{userA-id}
   ```

5. **Both users are now friends**
   - Request disappears from both lists
   - Both users see each other in friends list

---

## Common Mistakes to Avoid

### ❌ WRONG: Using /me/friend-requests for sent requests
```tsx
// This returns RECEIVED requests, not SENT
const sentRequests = await get('/api/users/me/friend-requests');
// sentRequests will NOT contain requests you sent!
```

### ✅ CORRECT: Use separate endpoints
```tsx
// For received requests
const received = await get('/api/users/me/friend-requests');

// For sent requests  
const sent = await get('/api/users/me/friend-requested');
```

---

## UI State Management

### After Sending Friend Request
```tsx
// User clicks "Add Friend"
await sendFriendRequest(userId);

// Option 1: Remove from suggestions immediately
setSuggestions(prev => prev.filter(u => u.id !== userId));

// Option 2: Update status to "SENT"
setSuggestions(prev => prev.map(u => 
  u.id === userId ? { ...u, requestStatus: 'SENT' } : u
));
```

### After Canceling Sent Request
```tsx
// User clicks "Cancel" on sent request
await cancelFriendRequest(userId);

// Remove from sent requests list
setSentRequests(prev => prev.filter(u => u.id !== userId));

// Add back to suggestions
refetchSuggestions();
```

---

## Testing Checklist

- [ ] Send friend request → appears in sender's SENT list
- [ ] Send friend request → appears in receiver's RECEIVED list
- [ ] Accept request → both become friends, request disappears
- [ ] Reject request → request disappears from both lists
- [ ] Cancel sent request → disappears from SENT list
- [ ] requestType is correct for all endpoints
- [ ] UI distinguishes between SENT and RECEIVED correctly
