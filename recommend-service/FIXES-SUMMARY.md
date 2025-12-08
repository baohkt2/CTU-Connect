# Recommendation Service Fixes Summary

## Overview
This document summarizes all fixes applied to the recommendation service to resolve Kafka message deserialization, Redis connection, and embedding generation issues.

## Issue #1: Kafka Message Conversion Error

### Problem
```
MessageConversionException: Cannot convert from [java.lang.String] to [vn.ctu.edu.recommend.kafka.event.PostEvent]
```

### Root Causes
1. **Structure Mismatch**: PostEvent class expected flat structure, but message had nested `data` object
2. **Deserializer Configuration**: Missing proper JSON deserializer configuration

### Solution
Created `KafkaConsumerConfig` with custom deserializers and updated `PostEvent` class structure.

**Files Changed:**
- `PostEvent.java` - Added nested data classes to match message structure
- `KafkaConsumerConfig.java` (NEW) - Custom Kafka consumer configuration
- `PostEventConsumer.java` - Updated @KafkaListener annotations
- `application.yml` - Added JSON type mapping properties

**Key Features:**
- ErrorHandlingDeserializer wraps JsonDeserializer
- USE_TYPE_INFO_HEADERS disabled
- Backward-compatible convenience methods
- @JsonIgnoreProperties for flexible deserialization

## Issue #2: Redis Connection Failure

### Problem
```
RedisConnectionFailureException: Unable to connect to Redis
Network is unreachable: localhost/0.0.0.1:6380
```

### Root Cause
Application tried resolving `localhost` to IPv6, resulting in malformed address `0.0.0.1:6380`

### Solution
Changed Redis host to explicit IPv4 address `127.0.0.1` and enhanced RedisConfig with custom connection factory.

**Files Changed:**
- `application.yml` - Changed Redis host default to 127.0.0.1
- `application-dev.yml` - Changed Redis host to 127.0.0.1
- `RedisConfig.java` - Added custom LettuceConnectionFactory

**Key Features:**
- Explicit IPv4 addressing
- Proper connection timeouts (3000ms)
- Auto-reconnect enabled
- Socket keep-alive enabled

## Issue #3: Zero Embeddings in Database

### Problem
All embeddings in `post_embeddings` table are zero vectors (768 dimensions of zeros)

### Root Cause
- EmbeddingService was calling PhoBERT service on port 8096 (not running)
- Fallback logic returned zero vectors silently instead of failing
- Python AI service on port 8000 was running but not being used

### Solution
Updated `EmbeddingService` to use Python AI service on port 8000 with `/embed/post` endpoint.

**Files Changed:**
- `EmbeddingService.java` - Switched from PhoBERT service to Python service

**Key Changes:**
- Changed service URL from `localhost:8096` to `localhost:8000`
- Changed endpoint from `/api/nlp/embed` to `/embed/post`
- Updated request format to match Python service API
- Removed fallback embedding - now throws exception on failure
- Added proper logging for debugging

## Build Status
✅ All changes compiled successfully with no errors
✅ Python service tested and working (returns 768-dim real vectors)

## Testing Recommendations

### 1. Kafka Consumer Test
Create a test post in post-service and verify:
- Message is successfully consumed
- Post embedding is created in PostgreSQL with **real vectors (not zeros)**
- Cache is properly invalidated

### 2. Redis Connection Test
Verify Redis operations:
- Cache writes succeed
- Cache reads succeed
- TTL configurations work correctly

### 3. Embedding Generation Test
```bash
# Test Python service directly
curl -X POST http://localhost:8000/embed/post \
  -H "Content-Type: application/json" \
  -d '{"post_id":"test","content":"Công nghệ thông tin","title":""}'
```

Verify response contains 768-dimensional vector with non-zero values.

### 4. Database Verification
```sql
-- Check latest embedding
SELECT post_id, 
       length(embedding_vector) as vector_length,
       substring(embedding_vector, 1, 100) as vector_preview
FROM post_embeddings
ORDER BY created_at DESC
LIMIT 5;
```

Expected: `vector_length` > 100 chars, `vector_preview` contains real numbers (not all zeros)

### 5. Clean Old Data
```sql
-- Remove old zero-vector embeddings
DELETE FROM post_embeddings 
WHERE embedding_vector LIKE '[0,0,0%';
```

## Configuration Summary

### Kafka Consumer
```yaml
spring:
  kafka:
    consumer:
      value-deserializer: ErrorHandlingDeserializer
      properties:
        spring.json.trusted.packages: "*"
        spring.json.use.type.info.headers: false
```

### Redis Connection
```yaml
spring:
  data:
    redis:
      host: 127.0.0.1
      port: 6380
      password: recommend_redis_pass
      timeout: 3000ms
```

### Python AI Service
```yaml
recommendation:
  python-service:
    url: http://localhost:8000
    timeout: 10000
    enabled: true
```

## Documentation
- See `KAFKA-FIX.md` for detailed Kafka fix documentation
- See `REDIS-FIX.md` for detailed Redis fix documentation
- See `EMBEDDING-FIX.md` for detailed embedding fix documentation

## Important Notes
1. **Python service must be running** on port 8000 before starting Java service
2. **Old zero-vector data** should be cleaned from database
3. **Restart required** for all changes to take effect

## Date
December 9, 2025 (Updated)
