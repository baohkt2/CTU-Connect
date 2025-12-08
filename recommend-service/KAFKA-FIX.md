# Kafka Message Conversion Fix

## Problem
The recommendation service was unable to deserialize Kafka messages from the `post_created` topic, resulting in:
```
MessageConversionException: Cannot convert from [java.lang.String] to [vn.ctu.edu.recommend.kafka.event.PostEvent]
```

## Root Cause
Two issues were identified:

1. **Structure Mismatch**: The incoming Kafka message had a nested format with a `data` object containing post details, while the PostEvent class expected a flat structure.

2. **Deserializer Configuration**: The default Kafka consumer configuration wasn't properly configured to deserialize JSON strings into PostEvent objects. The message was being received as a raw String instead of being converted to the PostEvent class.

### Incoming Message Structure:
```json
{
  "data": {
    "id": "69370d53eb61c718b78042fa",
    "title": "chào",
    "content": "CPU và GPU và Công nghệ",
    "author": {...},
    "category": "technology",
    "tags": [],
    ...
  },
  "eventType": "POST_CREATED",
  "postId": "69370d53eb61c718b78042fa",
  "authorId": "31ba8a23-8a4e-4b24-99c2-0d768e617e71",
  "timestamp": 1765215572005
}
```

## Solution
Applied a two-part fix:

### Part 1: Updated PostEvent Structure
Updated the `PostEvent` class to match the actual message structure:

1. **Created nested data classes** to match the incoming JSON structure
2. **Added convenience methods** (`getContent()`, `getCategory()`, `getTags()`) to maintain backward compatibility
3. **Added JSON annotations** (`@JsonIgnoreProperties(ignoreUnknown = true)`) to handle extra fields gracefully

### Part 2: Custom Kafka Consumer Configuration
Created a dedicated `KafkaConsumerConfig` class with proper deserializer setup:

1. **ErrorHandlingDeserializer** wrapper to handle deserialization errors gracefully
2. **JsonDeserializer** configured specifically for PostEvent class
3. **Disabled USE_TYPE_INFO_HEADERS** to ignore type headers from message
4. **Custom ConsumerFactory** with explicit deserializer configuration
5. **ConcurrentKafkaListenerContainerFactory** bean for listener methods

### Changes Made:

#### 1. PostEvent.java
- Added nested `PostData` class containing all post details
- Added nested classes: `Author`, `Stats`, `AudienceSettings`, `Engagement`
- Added convenience methods to access nested data without breaking existing code
- Added `@JsonIgnoreProperties(ignoreUnknown = true)` for flexible deserialization
- Changed `timestamp` from LocalDateTime to Long to match incoming format

#### 2. KafkaConsumerConfig.java (NEW)
```java
@EnableKafka
@Configuration
public class KafkaConsumerConfig {
    // Custom ConsumerFactory with ErrorHandlingDeserializer
    // JsonDeserializer configured for PostEvent.class
    // USE_TYPE_INFO_HEADERS set to false
    // TRUSTED_PACKAGES set to "*"
}
```

#### 3. PostEventConsumer.java
- Updated all `@KafkaListener` annotations to use `containerFactory = "kafkaListenerContainerFactory"`
- This ensures listeners use the custom deserializer configuration

#### 4. application.yml
- Added `spring.json.type.mapping` property
- Added `spring.json.value.default.type` property

## Key Configuration Properties
```yaml
spring.kafka.consumer:
  value-deserializer: ErrorHandlingDeserializer
  properties:
    spring.json.trusted.packages: "*"
    spring.json.use.type.info.headers: false
    spring.json.value.default.type: vn.ctu.edu.recommend.kafka.event.PostEvent
```

## Testing
Build completed successfully with no compilation errors. The service should now:
1. Properly deserialize JSON strings to PostEvent objects
2. Handle unknown fields gracefully
3. Support the nested message structure from post-service

## Backward Compatibility
The convenience methods ensure that existing consumer code like:
```java
event.getContent()    // Returns data.content
event.getCategory()   // Returns data.category
event.getTags()       // Returns data.tags as String[]
```
continues to work without modification.

## Date
December 9, 2025 (Updated)
