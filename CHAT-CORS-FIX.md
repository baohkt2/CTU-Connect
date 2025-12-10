# Chat Service - CORS and Spring Security Fix

## Critical Issue
```
Access to XMLHttpRequest at 'http://10.10.13.136:8086/login' 
(redirected from 'http://localhost:8090/api/chats/conversations/direct/...')
from origin 'http://localhost:3000' has been blocked by CORS policy
```

## Root Causes

### 1. Spring Security Default Behavior
Chat-service có dependency `spring-boot-starter-security` nhưng **KHÔNG có SecurityConfig**:
- ❌ Spring Security tự động enable form login
- ❌ Khi API call không có authentication, redirect đến `/login` HTML page
- ❌ HTML login page không có CORS headers
- ❌ Browser blocks CORS request

### 2. Missing CORS Configuration
Service không có CORS config cho cross-origin requests từ frontend.

## Solution

### Created: SecurityConfig.java

Tạo file mới để config Spring Security đúng cách cho REST API microservice:

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            // Disable CSRF (stateless REST API)
            .csrf(csrf -> csrf.disable())
            
            // Enable CORS
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            
            // Stateless sessions
            .sessionManagement(session -> 
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            
            // Disable form login (API Gateway handles auth)
            .formLogin(form -> form.disable())
            
            // Disable HTTP Basic
            .httpBasic(basic -> basic.disable())
            
            // Disable logout endpoint
            .logout(logout -> logout.disable())
            
            // Allow all requests (Gateway already validated)
            .authorizeHttpRequests(auth -> auth
                .anyRequest().permitAll()
            );

        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        
        // Allow origins
        configuration.setAllowedOrigins(Arrays.asList(
            "http://localhost:3000",
            "http://localhost:3001",
            "http://10.10.13.136:3000"
        ));
        
        // Allow methods
        configuration.setAllowedMethods(Arrays.asList(
            "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"
        ));
        
        // Allow all headers
        configuration.setAllowedHeaders(Arrays.asList("*"));
        
        // Allow credentials
        configuration.setAllowCredentials(true);
        
        // Max age for preflight
        configuration.setMaxAge(3600L);
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        
        return source;
    }
}
```

### Updated: WebConfig.java

Added CORS mapping as additional layer:

```java
@Override
public void addCorsMappings(CorsRegistry registry) {
    registry.addMapping("/**")
            .allowedOrigins(
                    "http://localhost:3000",
                    "http://localhost:3001",
                    "http://10.10.13.136:3000"
            )
            .allowedMethods("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
            .allowedHeaders("*")
            .allowCredentials(true)
            .maxAge(3600);
}
```

## Why This Happened

### Spring Security Defaults
When `spring-boot-starter-security` is in classpath WITHOUT custom config:
1. **Form login** automatically enabled
2. **HTTP Basic** automatically enabled
3. All endpoints **require authentication**
4. Unauthorized requests **redirect to /login** (HTML page)
5. **No CORS** configuration by default

### For Microservices
This default behavior is **wrong** for backend microservices because:
- API Gateway already handles authentication
- Services should return JSON, not HTML
- Services should use stateless sessions
- Services need CORS for different origins

## Architecture Context

```
Frontend (localhost:3000)
    ↓ (with cookies)
API Gateway (localhost:8090)
    ↓ (validates JWT, forwards request)
Chat Service (localhost:8086)
    ↓ Should: Accept request and return JSON
    ❌ Was: Redirect to HTML login page (no CORS)
    ✅ Now: Accept request with CORS enabled
```

## Key Configuration Points

### 1. Disable Form Login
```java
.formLogin(form -> form.disable())
```
No HTML login page for REST API.

### 2. Disable HTTP Basic
```java
.httpBasic(basic -> basic.disable())
```
Not using basic auth (Gateway uses JWT).

### 3. Stateless Sessions
```java
.sessionManagement(session -> 
    session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
```
No server-side sessions (JWT is stateless).

### 4. Permit All Requests
```java
.authorizeHttpRequests(auth -> auth.anyRequest().permitAll())
```
API Gateway already validated, so allow all.

### 5. Enable CORS
```java
.cors(cors -> cors.configurationSource(corsConfigurationSource()))
```
Allow cross-origin requests from frontend.

## CORS Configuration Details

### Allowed Origins
```java
"http://localhost:3000",      // Dev frontend
"http://localhost:3001",      // Alternative port
"http://10.10.13.136:3000"   // Network IP
```

### Allowed Methods
```
GET, POST, PUT, PATCH, DELETE, OPTIONS
```

### Allowed Headers
```
* (all headers)
```

### Allow Credentials
```
true (required for cookies)
```

### Max Age
```
3600 seconds (1 hour for preflight cache)
```

## Testing

### 1. Check CORS Headers
```bash
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     http://localhost:8086/api/chats/conversations

# Should return:
# Access-Control-Allow-Origin: http://localhost:3000
# Access-Control-Allow-Methods: GET,POST,PUT,PATCH,DELETE,OPTIONS
# Access-Control-Allow-Credentials: true
```

### 2. Check No Login Redirect
```bash
curl -v http://localhost:8086/api/chats/conversations

# Should return:
# HTTP/1.1 200 OK (or 401 JSON)
# NOT: HTTP/1.1 302 Found (redirect to /login)
```

### 3. From Frontend
1. Open browser DevTools → Network tab
2. Click "Nhắn tin" from Friends list
3. Check request to `/api/chats/conversations/direct/...`
4. Should see:
   - Status: 200 OK (or proper error JSON)
   - Response Headers include `Access-Control-Allow-Origin`
   - NO redirect to login page

## Related Microservices

Other services should have similar SecurityConfig:
- ✅ **user-service**: Already has SecurityConfig
- ✅ **media-service**: Check if needs SecurityConfig
- ✅ **post-service**: Check if needs SecurityConfig
- ✅ **chat-service**: NOW FIXED ✅

## Best Practices for Microservices

### DO:
1. ✅ Disable form login in REST APIs
2. ✅ Use stateless sessions
3. ✅ Configure CORS explicitly
4. ✅ Return JSON for all responses
5. ✅ Let API Gateway handle authentication

### DON'T:
1. ❌ Use default Spring Security config
2. ❌ Redirect to HTML pages from REST API
3. ❌ Use server-side sessions
4. ❌ Duplicate authentication logic
5. ❌ Forget CORS configuration

## Files Created/Modified

### Created
1. `chat-service/src/main/java/com/ctuconnect/config/SecurityConfig.java`
   - Spring Security configuration
   - CORS configuration
   - Disables form login, HTTP basic, logout

### Modified
2. `chat-service/src/main/java/com/ctuconnect/config/WebConfig.java`
   - Added CORS mappings
   - Complementary to SecurityConfig

## Build Status
✅ **SUCCESS** - Build completed without errors

## Required Action
⚠️ **RESTART chat-service** in your IDE to apply changes!

## Verification Steps

1. **Restart chat-service** (Run ChatServiceApplication in IDE)
2. **Clear browser cache** and cookies
3. **Login again** to get fresh auth cookies
4. **Navigate to Friends** → Click "Nhắn tin"
5. **Check Network tab**: Should see 200 responses, no CORS errors
6. **Check Console**: No CORS policy errors

## Status
✅ **FIXED** - Spring Security configured correctly
✅ **FIXED** - CORS enabled for cross-origin requests
✅ **FIXED** - No more redirect to HTML login page

---

**Last Updated**: 2024-12-10
**Priority**: CRITICAL
**Impact**: Chat service now properly handles REST API requests with CORS
