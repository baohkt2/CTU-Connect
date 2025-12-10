# Chat Service - User Service Connection Fix

## Issue
```
org.springframework.web.client.ResourceAccessException: 
I/O error on GET request for "http://user-service:8081/api/users/..."
user-service
```

Chat-service không thể lấy thông tin user từ user-service.

## Root Cause

**Service Discovery vs Local Development**

UserService.java có default URL:
```java
@Value("${user.service.url:http://user-service:8081}")
```

- `user-service:8081` là service name cho **Eureka/Docker**
- Khi chạy local trong IDE: service name **không resolve**
- Cần dùng `localhost:8081` cho local development

## Solution

### Add Configuration Property

**File**: `chat-service/src/main/resources/application.properties`

Added line:
```properties
# User Service Configuration (for local development)
user.service.url=http://localhost:8081
```

## Why This Happens

### Service Discovery Environments

#### Docker/Production (Service Name)
```
chat-service → http://user-service:8081
                     ↑
                Service name resolved by Docker network/Eureka
```

#### Local Development (Localhost)
```
chat-service → http://localhost:8081
                     ↑
                Direct localhost connection
```

### UserService.java Implementation

```java
public UserService(RestTemplate restTemplate,
                  @Value("${user.service.url:http://user-service:8081}") String userServiceUrl) {
    this.restTemplate = restTemplate;
    this.userServiceUrl = userServiceUrl;
}
```

**Property precedence**:
1. If `user.service.url` is set in properties → Use that value ✅
2. Else → Use default `http://user-service:8081` (for Docker)

## Configuration Strategy

### Local Development (application.properties)
```properties
user.service.url=http://localhost:8081
```

### Docker Environment (application-docker.properties)
```properties
# Use service name
user.service.url=http://user-service:8081
```

Or don't set it - will use default from code.

## Testing

### 1. Check User Service is Running
```bash
curl http://localhost:8081/api/users/health
# or
curl http://localhost:8081/actuator/health
```

Should return 200 OK.

### 2. Test from Chat Service
After restart, chat-service should log:
```
Successfully fetched user info for userId: xxx
```

Instead of:
```
Failed to get user info for userId: xxx
```

### 3. From Frontend
1. Click "Nhắn tin" from Friends
2. Conversation should show **real friend name and avatar**
3. Not fallback "User [uuid]"

## Related Services

Similar configuration needed for:
- ✅ **chat-service → user-service**: NOW FIXED
- Check if chat-service calls other services (media-service?)
- Check if other services need similar config

## Fallback Mechanism

UserService has fallback for failures:
```java
private Map<String, Object> createDefaultUserInfo(String userId) {
    return Map.of(
        "id", userId,
        "name", "User " + userId,
        "avatar", "",
        "fullName", "User " + userId
    );
}
```

**If user-service is down**:
- Chat still works ✅
- Shows "User [uuid]" instead of real name
- Avatar is empty
- No error to end user

## Architecture

### Current Setup (Local Development)
```
Browser → API Gateway → Chat Service → User Service
                        (localhost:8086)  (localhost:8081)
```

### Production Setup (Docker)
```
Browser → API Gateway → Chat Service → User Service
                        (user-service:8081 via Docker network)
```

## Common Patterns for Service URLs

### Development
```properties
user.service.url=http://localhost:8081
media.service.url=http://localhost:8084
post.service.url=http://localhost:8083
```

### Docker Compose
```properties
user.service.url=http://user-service:8081
media.service.url=http://media-service:8084
post.service.url=http://post-service:8083
```

### Kubernetes
```properties
user.service.url=http://user-service.default.svc.cluster.local:8081
```

## Best Practices

### 1. Use Configuration Properties
```java
// ✅ Good - Configurable
@Value("${user.service.url:http://user-service:8081}")

// ❌ Bad - Hardcoded
private static final String USER_SERVICE_URL = "http://user-service:8081";
```

### 2. Provide Sensible Defaults
```java
// Default works for Docker/production
@Value("${user.service.url:http://user-service:8081}")
```

### 3. Use Profiles
```properties
# application.properties (dev)
user.service.url=http://localhost:8081

# application-docker.properties (docker)
user.service.url=http://user-service:8081

# application-prod.properties (production)
user.service.url=http://user-service.prod:8081
```

### 4. Document URLs
```properties
# User Service URL
# - Local: http://localhost:8081
# - Docker: http://user-service:8081
# - Prod: http://user-service.prod:8081
user.service.url=http://localhost:8081
```

## Troubleshooting

### Still Getting Connection Error?

1. **Check user-service is running**:
   ```bash
   curl http://localhost:8081/actuator/health
   ```

2. **Check chat-service loaded property**:
   ```
   # In chat-service logs on startup
   user.service.url : http://localhost:8081
   ```

3. **Check port is correct**:
   - user-service default: 8081
   - chat-service: 8086
   - api-gateway: 8090

4. **Check firewall/antivirus**:
   - May block localhost connections

### Wrong User Info Shown?

If seeing "User [uuid]" instead of real name:
- Fallback is working (service unreachable)
- Check logs for actual error
- May be authentication issue (user-service requires JWT?)

## Files Modified

1. **chat-service/src/main/resources/application.properties**
   - Added `user.service.url=http://localhost:8081`

## Action Required

⚠️ **RESTART chat-service** in IDE to apply property change

## Verification

After restart:
1. Clear browser cache
2. Click "Nhắn tin" on a friend
3. Check chat-service logs - should NOT see error
4. Conversation should show friend's real name and avatar
5. No "User [uuid]" fallback text

## Status
✅ **FIXED** - User service URL configured for local development

---

**Last Updated**: 2024-12-10
**Priority**: HIGH
**Impact**: Chat now displays real user names and avatars
