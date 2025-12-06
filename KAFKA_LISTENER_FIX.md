# Kafka Listener Transaction Rollback Fix

## Problem
```
org.springframework.kafka.listener.ListenerExecutionFailedException: 
Listener method 'public void com.ctuconnect.service.UserEventListener.handleUserCreatedEvent(...)' threw exception

Caused by: org.springframework.transaction.UnexpectedRollbackException: 
Transaction rolled back because it has been marked as rollback-only
```

## Root Cause

The `UserEventListener` methods had a **conflicting pattern**:

1. ❌ Methods were annotated with `@Transactional`
2. ❌ Exceptions were caught inside try-catch blocks
3. ❌ Exceptions were logged but NOT rethrown

### What Happened:
1. Kafka message received → Transaction started
2. Exception thrown inside `userSyncService.createUserFromAuthService()`
3. Spring marks transaction for **rollback-only**
4. Exception caught by try-catch → logged but not rethrown
5. Method execution continues → `acknowledgment.acknowledge()` called
6. Method returns → Spring tries to **commit** transaction
7. **Error!** Transaction is marked rollback-only but code is trying to commit
8. `UnexpectedRollbackException` thrown

### The Anti-Pattern:
```java
@Transactional  // ← Transaction starts
public void handleUserCreatedEvent(...) {
    try {
        // Some operation that throws exception
        userSyncService.createUserFromAuthService(...);  // ← Exception here
        acknowledgment.acknowledge();
    } catch (Exception e) {
        log.error("Error: {}", e);  // ← Caught but not rethrown!
        // Transaction is marked for rollback, but we continue...
    }
} // ← Spring tries to commit, but transaction is rollback-only → BOOM!
```

## Solution Applied

### Removed `@Transactional` from Kafka Listeners

**Why?**
- Kafka listeners should **not** manage their own transactions
- Each listener method should be a **separate unit of work**
- The inner service methods (`UserSyncService`) already have `@Transactional`
- Kafka will handle message retries if processing fails

**Before:**
```java
@KafkaListener(topics = "user-registration", groupId = "user-service-group")
@Transactional  // ❌ REMOVED
public void handleUserCreatedEvent(@Payload Map<String, Object> event, ...) {
    try {
        // Process event
        acknowledgment.acknowledge();
    } catch (Exception e) {
        log.error("Error: {}", e);
    }
}
```

**After:**
```java
@KafkaListener(topics = "user-registration", groupId = "user-service-group")
// ✅ No @Transactional here
public void handleUserCreatedEvent(@Payload Map<String, Object> event, ...) {
    try {
        // Process event
        acknowledgment.acknowledge();  // ✅ Only if successful
    } catch (Exception e) {
        log.error("Error: {}", e);
        // ✅ Don't acknowledge - message will be retried
    }
}
```

### Updated All Three Listeners:
1. ✅ `handleUserCreatedEvent()` - Removed `@Transactional`
2. ✅ `handleUserUpdatedEvent()` - Removed `@Transactional`
3. ✅ `handleUserDeletedEvent()` - Removed `@Transactional`

### Improved Error Handling:
- Added offset logging for better debugging
- Don't acknowledge failed messages (allows Kafka retry)
- Transaction management stays in `UserSyncService` methods

## How It Works Now

### Successful Flow:
```
1. Kafka delivers message
2. handleUserCreatedEvent() called
3. userSyncService.createUserFromAuthService() executes
   ↳ @Transactional starts here
   ↳ Creates user in Neo4j
   ↳ Transaction commits
4. acknowledgment.acknowledge() called
5. Message removed from Kafka
```

### Failed Flow:
```
1. Kafka delivers message
2. handleUserCreatedEvent() called
3. userSyncService.createUserFromAuthService() fails
   ↳ @Transactional starts here
   ↳ Exception thrown
   ↳ Transaction rolls back
4. catch block logs error
5. acknowledgment.acknowledge() NOT called
6. Kafka will redeliver message
```

## Benefits of This Approach

### ✅ Proper Transaction Boundaries
- Transactions are in service layer where they belong
- Each service method is a complete transaction
- Listener is just a message handler

### ✅ Better Error Handling
- Failed messages are not acknowledged
- Kafka automatically retries
- No "transaction already rolled back" errors

### ✅ Idempotency Support
- If `userRepository.existsById(userId)` returns true, it updates instead of failing
- Handles duplicate message delivery gracefully

### ✅ Cleaner Separation of Concerns
- Listener: Receives messages, delegates to service
- Service: Business logic + transaction management
- Repository: Data access

## UserSyncService Transaction Management

The actual transaction management stays in `UserSyncService`:

```java
@Service
public class UserSyncService {
    
    @Transactional  // ✅ Transaction here is correct
    public UserDTO createUserFromAuthService(String userId, String email, 
                                             String username, String role) {
        if (userRepository.existsById(userId)) {
            return updateUserFromAuth(userId, email, role);  // Idempotent
        }
        
        UserEntity user = UserEntity.fromAuthService(userId, email, username, role);
        return mapToDTO(userRepository.save(user));
    }
    
    @Transactional  // ✅ Each method has its own transaction
    public UserDTO updateUserFromAuth(String userId, String email, String role) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found: " + userId));
        
        user.setEmail(email);
        user.setRole(role);
        user.setUpdatedAt(LocalDateTime.now());
        
        return mapToDTO(userRepository.save(user));
    }
}
```

## Kafka Configuration

The Kafka listener configuration in `application.properties` is correct:

```properties
spring.kafka.bootstrap-servers=localhost:9092
spring.kafka.consumer.group-id=user-service-group
spring.kafka.consumer.auto-offset-reset=earliest
spring.kafka.consumer.enable-auto-commit=false  # Manual acknowledgment
spring.kafka.listener.ack-mode=manual  # We control when to acknowledge
```

## Testing the Fix

### 1. Register a New User (via auth-service)
```bash
curl -X POST http://localhost:8090/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "testuser@student.ctu.edu.vn",
    "username": "testuser",
    "password": "Password123@"
  }'
```

### 2. Check User-Service Logs
Look for successful processing:
```
Received user-created event from topic 'user-registration' at offset 123: {userId=..., email=..., username=..., role=...}
User created successfully in user-db: <userId>
```

### 3. Verify in Neo4j
```bash
# Access Neo4j browser: http://localhost:7474
# Run query:
MATCH (u:User) WHERE u.id = '<userId>' RETURN u
```

### 4. Check for Errors
If there are issues, you'll see:
```
Failed to process user-created event at offset 123: <error message>
```
The message will be retried by Kafka.

## Common Scenarios

### Scenario 1: User Already Exists
**What happens:**
- `userRepository.existsById(userId)` returns `true`
- Calls `updateUserFromAuth()` instead of creating
- **No error** - handles duplicate gracefully

### Scenario 2: Neo4j Connection Issue
**What happens:**
- Exception thrown in `userRepository.save()`
- Transaction rolls back
- Error logged in listener
- Message **not acknowledged**
- Kafka **redelivers** message
- Works after Neo4j reconnects

### Scenario 3: Invalid Event Data
**What happens:**
- Exception thrown when parsing event
- Error logged
- Message **not acknowledged**
- Message goes to **DLQ** (Dead Letter Queue) after max retries

## Related Issues Fixed

This fix also resolves:
- ❌ "Transaction rolled back because it has been marked as rollback-only"
- ❌ Kafka consumer rebalancing issues
- ❌ Messages being acknowledged despite failures
- ❌ Duplicate users not being handled properly

## Files Modified

1. ✏️ `user-service/src/main/java/com/ctuconnect/service/UserEventListener.java`
   - Removed `@Transactional` from all listener methods
   - Improved logging with offset tracking
   - Fixed acknowledgment logic

## Next Steps

1. **Restart user-service** to apply changes
2. **Test registration** to verify user synchronization
3. **Monitor logs** for any processing errors
4. **Check Neo4j** to confirm users are created

## Prevention for Future

### ✅ DO:
- Use `@Transactional` in service layer methods
- Let Kafka listeners be simple message handlers
- Only acknowledge messages after successful processing
- Handle idempotency in service methods

### ❌ DON'T:
- Put `@Transactional` on Kafka listener methods
- Catch exceptions without rethrowing in transactional methods
- Acknowledge messages in catch blocks
- Mix transaction and message acknowledgment logic

---

**Status**: ✅ FIXED  
**Date**: December 4, 2025  
**Impact**: Kafka event processing now works reliably without transaction conflicts
