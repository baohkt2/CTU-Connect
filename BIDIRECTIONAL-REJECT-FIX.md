# Bidirectional Reject/Cancel Fix - December 10, 2025

## Problem
Người gửi không thể hủy lời mời kết bạn đã gửi vì `rejectFriendRequest` chỉ hoạt động 1 chiều (requester → rejecter).

## Solution: Bidirectional Reject/Cancel

### Query mới hỗ trợ 2 chiều:

```cypher
MATCH (user1:User {id: $userId1})
MATCH (user2:User {id: $userId2})
OPTIONAL MATCH (user1)-[r1:SENT_FRIEND_REQUEST_TO]->(user2)
OPTIONAL MATCH (user2)-[r2:SENT_FRIEND_REQUEST_TO]->(user1)
WITH r1, r2
WHERE r1 IS NOT NULL OR r2 IS NOT NULL
DELETE r1, r2
RETURN count(r1) + count(r2) > 0 as success
```

### Logic:
1. Tìm user1 và user2
2. Tìm relationship theo CẢ 2 chiều:
   - `user1 → user2` (r1)
   - `user2 → user1` (r2)
3. Nếu tồn tại relationship nào → Delete
4. Return success nếu đã xóa được

### Use Cases:

**Case 1: Người nhận reject**
```
User A sends request to User B
User B calls: rejectFriendRequest(A, B)
→ Finds A→B relationship → Deletes → Success
```

**Case 2: Người gửi cancel**
```
User A sends request to User B
User A calls: rejectFriendRequest(A, B)
→ Finds A→B relationship → Deletes → Success
```

**Case 3: Bất kể thứ tự parameters**
```
User A sends request to User B

User B calls: rejectFriendRequest(B, A)  
→ Finds A→B relationship → Deletes → Success

User A calls: rejectFriendRequest(B, A)
→ Finds A→B relationship → Deletes → Success
```

## API Unchanged

Endpoints vẫn giữ nguyên, chỉ logic backend thay đổi:

```bash
# User B rejects request from User A
POST /api/users/me/reject-invite/{userA-id}

# User A cancels request to User B  
POST /api/users/me/reject-invite/{userB-id}

# Both use the same endpoint!
```

## Frontend Code

```typescript
// Reject received request
const handleRejectRequest = async (friendId: string) => {
  await userService.rejectFriendRequest(friendId);
  // Works!
};

// Cancel sent request
const handleCancelRequest = async (friendId: string) => {
  await userService.cancelFriendRequest(friendId);
  // Also works! (calls same API)
};
```

## Benefits

✅ **Đơn giản hóa logic** - 1 method cho cả reject và cancel  
✅ **Không cần phân biệt** ai gửi, ai nhận  
✅ **Flexible** - Frontend có thể gọi với bất kỳ thứ tự parameters  
✅ **Safer** - Tìm relationship theo cả 2 chiều  

## Testing

### Test 1: Reject received request
```bash
# User A sends to User B
POST /api/users/me/invite/userB-id  # From A

# User B rejects
POST /api/users/me/reject-invite/userA-id  # From B
→ ✅ Request deleted
```

### Test 2: Cancel sent request
```bash
# User A sends to User B
POST /api/users/me/invite/userB-id  # From A

# User A cancels
POST /api/users/me/reject-invite/userB-id  # From A
→ ✅ Request deleted
```

### Test 3: Non-existent request
```bash
# No request exists
POST /api/users/me/reject-invite/userC-id
→ ❌ Error: "Unable to reject/cancel friend request"
```

## Files Changed

1. **UserRepository.java** - Updated `rejectFriendRequest` query
2. **UserService.java** - Updated method comments and logs

## Summary

**Một method, hai chức năng:**
- 🔴 Người nhận → Reject
- 🔵 Người gửi → Cancel

**Cả hai đều gọi cùng 1 API endpoint!** 🎉
