# Redis Connection Fix

## Problem
The recommendation service was unable to connect to Redis, resulting in:
```
RedisConnectionFailureException: Unable to connect to Redis
Caused by: Network is unreachable: no further information: localhost/0.0.0.1:6380
```

## Root Cause
The application was trying to resolve `localhost` to both IPv4 and IPv6 addresses. Spring Boot's Lettuce Redis client attempted to connect via IPv6 first, which resulted in an invalid address `0.0.0.1:6380` (malformed IPv6). 

While Redis was correctly listening on:
- IPv4: `0.0.0.0:6380` (all interfaces)
- IPv6: `[::]:6380` (all interfaces)
- IPv6 loopback: `[::1]:6380`

The client was generating an incorrect IPv6 address format when trying to resolve `localhost`.

## Solution
Applied a two-part fix to enforce IPv4 connections:

### Part 1: Updated Configuration Files
Changed Redis host from `localhost` to explicit IPv4 address `127.0.0.1` in configuration files.

#### application.yml
```yaml
spring:
  data:
    redis:
      host: ${REDIS_HOST:127.0.0.1}  # Changed from localhost
      port: ${REDIS_PORT:6380}
      password: ${REDIS_PASSWORD:recommend_redis_pass}
```

#### application-dev.yml
```yaml
spring:
  data:
    redis:
      host: 127.0.0.1  # Changed from localhost
      port: 6380
      password: recommend_redis_pass
```

### Part 2: Enhanced RedisConfig
Updated the `RedisConfig` class to create a custom `LettuceConnectionFactory` with proper connection settings:

```java
@Bean
public LettuceConnectionFactory redisConnectionFactory() {
    // Redis standalone configuration with explicit IPv4
    RedisStandaloneConfiguration redisConfig = new RedisStandaloneConfiguration();
    redisConfig.setHostName(redisHost);  // Uses 127.0.0.1
    redisConfig.setPort(redisPort);
    redisConfig.setPassword(redisPassword);
    redisConfig.setDatabase(redisDatabase);

    // Socket options with proper timeouts
    SocketOptions socketOptions = SocketOptions.builder()
        .connectTimeout(redisTimeout)
        .keepAlive(true)
        .build();

    // Client options with auto-reconnect
    ClientOptions clientOptions = ClientOptions.builder()
        .socketOptions(socketOptions)
        .autoReconnect(true)
        .build();

    // Lettuce client configuration
    LettuceClientConfiguration clientConfig = LettuceClientConfiguration.builder()
        .clientOptions(clientOptions)
        .commandTimeout(redisTimeout)
        .build();

    return new LettuceConnectionFactory(redisConfig, clientConfig);
}
```

## Key Configuration Features

### 1. Explicit IPv4 Address
- Uses `127.0.0.1` instead of `localhost` to avoid DNS resolution issues
- Ensures consistent IPv4-only connections

### 2. Proper Timeouts
- Connect timeout: 3000ms (configurable via `spring.data.redis.timeout`)
- Command timeout: 3000ms
- Prevents hanging connections

### 3. Auto-Reconnect
- Enabled auto-reconnect in ClientOptions
- Ensures resilience to temporary network issues

### 4. Keep-Alive
- Socket keep-alive enabled
- Maintains persistent connections

## Changes Made

### Files Modified:
1. **application.yml** - Changed default Redis host to 127.0.0.1
2. **application-dev.yml** - Changed Redis host to 127.0.0.1
3. **RedisConfig.java** - Added custom LettuceConnectionFactory bean with proper configuration

## Testing
Build completed successfully. Connection test to 127.0.0.1:6380 verified:
```
TcpTestSucceeded : True
```

## Benefits
- Eliminates IPv6 resolution issues on Windows environments
- More reliable connection handling with explicit timeouts
- Auto-reconnection support for improved resilience
- Consistent behavior across different network configurations

## Docker Environment
Note: In Docker environments (application-docker.yml), the service uses hostname `recommend-redis` which is resolved by Docker's internal DNS. This configuration remains unchanged as Docker handles name resolution properly.

## Date
December 9, 2025
