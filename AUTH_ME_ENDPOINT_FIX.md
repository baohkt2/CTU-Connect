# Fix cho endpoint /api/auth/me bị lỗi 401 UNAUTHORIZED

## Vấn đề
- Endpoint `/api/auth/me` trả về 401 UNAUTHORIZED dù người dùng đã đăng nhập thành công
- Access token được lưu trong cookie nhưng không được xác thực đúng cách

## Nguyên nhân
1. **API Gateway**: Route `/api/auth/**` không áp dụng JWT filter
2. **Auth Service**: Endpoint `/me` cố gắng đọc access token trực tiếp từ cookie, nhưng cookie không được forward/validate đúng cách
3. Thiếu cơ chế forward thông tin user từ gateway xuống auth-service sau khi validate token

## Giải pháp
### 1. Cập nhật RouteConfig.java (API Gateway)
- Áp dụng JWT filter cho route `/api/auth/**`
- Filter sẽ validate token và skip các endpoint public (login, register, etc.)
- Thêm user headers (X-User-Id, X-User-Email) sau khi validate token

**File**: `api-gateway/src/main/java/com/ctuconnect/config/RouteConfig.java`

```java
// Auth Service Routes - Apply JWT filter (will skip public endpoints internally)
.route("auth-service-route", r -> r
        .path("/api/auth/**")
        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
        .uri("lb://auth-service"))
```

### 2. Cập nhật AuthController.java (Auth Service)
- Thêm logic đọc user info từ headers (X-User-Id, X-User-Email) được forward từ gateway
- Giữ fallback đọc từ cookie để backward compatibility

**File**: `auth-service/src/main/java/com/ctuconnect/controller/AuthController.java`

```java
@GetMapping("/me")
public ResponseEntity<AuthResponse> getCurrentUser(
        @RequestHeader(value = "X-User-Id", required = false) String userId,
        @RequestHeader(value = "X-User-Email", required = false) String userEmail,
        @CookieValue(value = "accessToken", required = false) String accessToken) {
    
    // If headers are present, use them (token was validated by gateway)
    if (userId != null && !userId.trim().isEmpty()) {
        try {
            AuthResponse authResponse = authService.getCurrentUserByEmail(userEmail != null ? userEmail : userId);
            return ResponseEntity.ok(authResponse);
        } catch (Exception e) {
            return ResponseEntity.status(401).build();
        }
    }
    
    // Fallback to cookie-based auth (for backward compatibility)
    if (accessToken == null || accessToken.trim().isEmpty()) {
        return ResponseEntity.status(401).build();
    }

    try {
        AuthResponse authResponse = authService.getCurrentUser(accessToken);
        return ResponseEntity.ok(authResponse);
    } catch (Exception e) {
        return ResponseEntity.status(401).build();
    }
}
```

### 3. Thêm method getCurrentUserByEmail
- Thêm interface method trong `AuthService.java`
- Implement trong `AuthServiceImpl.java`

**File**: `auth-service/src/main/java/com/ctuconnect/service/AuthService.java`
```java
AuthResponse getCurrentUserByEmail(String email);
```

**File**: `auth-service/src/main/java/com/ctuconnect/service/impl/AuthServiceImpl.java`
```java
@Override
public AuthResponse getCurrentUserByEmail(String email) {
    String normalizedEmail = email.toLowerCase().trim();
    UserEntity user = userRepository.findByEmail(normalizedEmail)
            .orElseThrow(() -> new RuntimeException("User not found"));

    if (!user.isActive()) {
        throw new RuntimeException("User account is inactive");
    }

    return AuthResponse.builder()
            .user(UserMapper.toDto(user))
            .tokenType("Bearer")
            .build();
}
```

## Luồng xử lý sau khi fix
1. Client gửi request đến `/api/auth/me` với access token trong cookie
2. API Gateway nhận request, áp dụng JwtAuthenticationFilter
3. Filter kiểm tra xem `/api/auth/me` có phải open endpoint không → Không
4. Filter extract access token từ cookie
5. Filter validate token → OK
6. Filter thêm headers: X-User-Id, X-User-Email
7. Gateway forward request + headers xuống auth-service
8. AuthController đọc headers X-User-Id, X-User-Email
9. Gọi authService.getCurrentUserByEmail() với email từ header
10. Trả về user info

## Files đã thay đổi
- `api-gateway/src/main/java/com/ctuconnect/config/RouteConfig.java`
- `auth-service/src/main/java/com/ctuconnect/controller/AuthController.java`
- `auth-service/src/main/java/com/ctuconnect/service/AuthService.java`
- `auth-service/src/main/java/com/ctuconnect/service/impl/AuthServiceImpl.java`

## Testing
Rebuild và restart các services:
```bash
cd api-gateway
mvn clean package -DskipTests
docker-compose up -d --build api-gateway

cd ../auth-service
mvn clean package -DskipTests
docker-compose up -d --build auth-service
```

Test endpoint:
```bash
# 1. Login
curl -c cookies.txt -X POST http://localhost:8090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@student.ctu.edu.vn","password":"password123"}'

# 2. Get current user (should return 200 OK with user info)
curl -b cookies.txt http://localhost:8090/api/auth/me
```
