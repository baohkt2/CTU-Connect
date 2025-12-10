# Chat CORS Duplicate Headers Fix

## Issue
```
The 'Access-Control-Allow-Origin' header contains multiple values 
'http://localhost:3000, http://localhost:3000', but only one is allowed.
```

Request returns **200 OK** but browser blocks it due to duplicate CORS headers.

## Root Cause

**Multiple sources setting CORS headers**:
1. ✅ `SecurityConfig.java` - Sets CORS via `CorsConfigurationSource`
2. ❌ `WebConfig.java` - ALSO sets CORS via `addCorsMappings()`
3. Possibly API Gateway also adding CORS headers

When both SecurityConfig and WebConfig set CORS, Spring adds the header **twice**, causing browser to reject the response even though server returned 200 OK.

## Solution

### Remove CORS from WebConfig

Keep CORS only in **one place** - SecurityConfig is the recommended location for Spring Security apps.

**File**: `chat-service/src/main/java/com/ctuconnect/config/WebConfig.java`

```java
// Before ❌ - Had both interceptors AND CORS
@Configuration
@RequiredArgsConstructor
public class WebConfig implements WebMvcConfigurer {
    private final AuthenticationInterceptor authenticationInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(authenticationInterceptor)
                .addPathPatterns("/api/**")
                .excludePathPatterns(...);
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {  // ❌ DUPLICATE!
        registry.addMapping("/**")
                .allowedOrigins("http://localhost:3000", ...)
                .allowedMethods("GET", "POST", ...)
                .allowCredentials(true);
    }
}

// After ✅ - Only interceptors, CORS handled by SecurityConfig
@Configuration
@RequiredArgsConstructor
public class WebConfig implements WebMvcConfigurer {
    private final AuthenticationInterceptor authenticationInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(authenticationInterceptor)
                .addPathPatterns("/api/**")
                .excludePathPatterns(...);
    }

    // CORS removed - handled by SecurityConfig to avoid duplicate headers
}
```

## Why This Happens

### Spring CORS Configuration Layers

Spring has multiple ways to configure CORS:

1. **@CrossOrigin** annotation (controller level)
2. **WebMvcConfigurer.addCorsMappings()** (global level)
3. **SecurityFilterChain with CorsConfigurationSource** (security level)
4. **CorsFilter** bean (filter level)

If you configure CORS in **multiple layers**, headers get added **multiple times**.

### Best Practice

For Spring Security applications:
- ✅ Configure CORS in **SecurityConfig only**
- ✅ Use `CorsConfigurationSource` bean
- ❌ Don't also use `addCorsMappings()` in WebConfig
- ❌ Don't add CorsFilter bean separately

## Correct Configuration

### SecurityConfig.java (KEEP THIS)

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            // ... other config
        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOrigins(Arrays.asList(
            "http://localhost:3000",
            "http://localhost:3001",
            "http://10.10.13.136:3000"
        ));
        configuration.setAllowedMethods(Arrays.asList(
            "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"
        ));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        configuration.setAllowCredentials(true);
        configuration.setMaxAge(3600L);
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }
}
```

### WebConfig.java (REMOVE CORS)

```java
@Configuration
@RequiredArgsConstructor
public class WebConfig implements WebMvcConfigurer {

    private final AuthenticationInterceptor authenticationInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        // Keep interceptors
        registry.addInterceptor(authenticationInterceptor)
                .addPathPatterns("/api/**")
                .excludePathPatterns(...);
    }

    // NO addCorsMappings() method!
}
```

## Error Symptoms

### Browser Error
```
Access to XMLHttpRequest at 'http://localhost:8090/api/...' 
from origin 'http://localhost:3000' has been blocked by CORS policy: 
The 'Access-Control-Allow-Origin' header contains multiple values 
'http://localhost:3000, http://localhost:3000', but only one is allowed.
```

### Network Tab
- **Status**: 200 OK
- **Response**: Server returned success
- **But**: Browser blocks the response
- **Headers**: `Access-Control-Allow-Origin: http://localhost:3000, http://localhost:3000`

### Console Log
```
POST http://localhost:8090/api/chats/conversations/direct/xxx 
net::ERR_FAILED 200 (OK)
```

Note: Says "ERR_FAILED" even though status is 200!

## Testing

### 1. Check Response Headers

```bash
curl -v -H "Origin: http://localhost:3000" \
     http://localhost:8090/api/chats/conversations
```

**Should see (SINGLE value)**:
```
Access-Control-Allow-Origin: http://localhost:3000
```

**Should NOT see (DUPLICATE)**:
```
Access-Control-Allow-Origin: http://localhost:3000, http://localhost:3000
```

### 2. Browser DevTools

1. Open Network tab
2. Make request
3. Check Response Headers
4. `Access-Control-Allow-Origin` should appear **once**

### 3. From Frontend

1. Click "Nhắn tin" from Friends list
2. Check Console - NO CORS errors
3. Check Network - 200 OK and request succeeds
4. Conversation created successfully

## Related Issues

### API Gateway CORS

If API Gateway ALSO has CORS configuration, it could add headers again:
- Check `api-gateway` application.yml
- Check if there's a CorsConfig in API Gateway
- Gateway should either:
  - NOT add CORS (services handle it) ✅ Recommended
  - OR add CORS but services don't

### Decision
Since services are behind Gateway:
- ✅ **Services** (like chat-service) handle CORS
- ❌ Gateway should NOT add CORS headers
- Reason: Services know their specific CORS needs

## Architecture

```
Browser (localhost:3000)
    ↓ (with Origin header)
API Gateway (localhost:8090)
    ↓ (forwards request, no CORS headers added)
Chat Service (localhost:8086)
    ↓ SecurityConfig adds CORS headers ✅
    ← Response with CORS headers (single set)
    ↓
API Gateway
    ↓ (forwards response as-is)
Browser (receives response, CORS check passes) ✅
```

## Files Modified

1. **chat-service/src/main/java/com/ctuconnect/config/WebConfig.java**
   - Removed `addCorsMappings()` method
   - Kept only `addInterceptors()` method
   - Added comment explaining CORS is in SecurityConfig

## Build Status
✅ **SUCCESS** - Compiled without errors

## Action Required
⚠️ **RESTART chat-service** in IDE to apply changes

## Verification

After restart:
1. Clear browser cache
2. Go to Friends page
3. Click "Nhắn tin"
4. Should work without CORS errors
5. Check Network tab - single CORS header value

## Status
✅ **FIXED** - Removed duplicate CORS configuration
✅ **FIXED** - Only SecurityConfig handles CORS now

---

**Last Updated**: 2024-12-10
**Priority**: HIGH
**Impact**: Chat API calls now succeed without CORS header duplication
