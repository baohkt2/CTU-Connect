# User Interaction & Post Deletion Fix

## 🐛 Issues Fixed

### 1. User Interactions Not Being Recorded ❌
**Symptom**: User actions (like, comment, share) không được ghi nhận vào recommend-service

**Root Cause**: 
- post-service gửi `Map<String, Object>` qua Kafka
- recommend-service expect `UserActionEvent` object
- Kafka deserializer không thể convert → events bị reject

### 2. Post Deletion Not Working Properly ❌
**Symptom**: Xóa post không clean up data ở recommend-service

**Root Cause**:
- Listener tồn tại nhưng thiếu delete user_feedback
- Không có proper logging để debug

## ✅ Solutions Applied

### Fix 1: Standardize User Action Events

#### post-service/EventService.java

**Before** ❌:
```java
public void publishInteractionEvent(String postId, String userId, String interactionType) {
    Map<String, Object> event = new HashMap<>();
    event.put("eventType", "INTERACTION");
    event.put("postId", postId);
    event.put("userId", userId);
    event.put("interactionType", interactionType);
    event.put("actionType", interactionType); 
    event.put("timestamp", System.currentTimeMillis()); // Long timestamp
    
    kafkaTemplate.send("user_action", event);
}
```

**After** ✅:
```java
public void publishInteractionEvent(String postId, String userId, String interactionType) {
    // Create structured event matching UserActionEvent in recommend-service
    Map<String, Object> event = new HashMap<>();
    event.put("actionType", interactionType.toUpperCase()); // LIKE, COMMENT, SHARE
    event.put("userId", userId);
    event.put("postId", postId);
    event.put("timestamp", LocalDateTime.now().toString()); // ISO format
    event.put("metadata", Map.of(
        "source", "post-service",
        "eventTime", System.currentTimeMillis()
    ));
    
    kafkaTemplate.send("user_action", event);
    
    System.out.println("📤 Published user_action event: " + interactionType);
}
```

**Benefits**:
- Consistent field names
- Proper timestamp format
- Metadata for debugging
- Console logging for verification

#### recommend-service/UserActionConsumer.java

**Before** ❌:
```java
@KafkaListener(topics = "user_action", groupId = "recommendation-service-group")
public void handleUserAction(Object eventObject) {
    // Expects Object but Spring passes ConsumerRecord
    // Error: "No serializer found for class ConsumerRecord"
}
```

**After** ✅:
```java
@KafkaListener(
    topics = "user_action", 
    groupId = "recommendation-service-group",
    containerFactory = "userActionKafkaListenerContainerFactory"
)
public void handleUserAction(@Payload Object eventObject) {
    try {
        // @Payload extracts message payload from ConsumerRecord ✅
        log.debug("📨 Raw event object type: {}", eventObject.getClass().getName());
        // Should show: java.util.LinkedHashMap
        
        UserActionEvent event = parseUserActionEvent(eventObject);
        
        if (event == null || event.getActionType() == null) {
            log.warn("❌ Invalid user_action event");
            return;
        }
        
        log.info("📥 Received user_action: {} by user {} on post {}", 
            event.getActionType(), event.getUserId(), event.getPostId());
        
        // Process event...
    }
}

private UserActionEvent parseUserActionEvent(Object eventObject) {
    if (eventObject instanceof UserActionEvent) {
        return (UserActionEvent) eventObject;
    }
    
    if (eventObject instanceof Map) {
        Map<String, Object> map = (Map<String, Object>) eventObject;
        
        UserActionEvent event = new UserActionEvent();
        event.setActionType(getStringValue(map, "actionType"));
        event.setUserId(getStringValue(map, "userId"));
        event.setPostId(getStringValue(map, "postId"));
        event.setMetadata(map.get("metadata"));
        
        // Parse timestamp from various formats
        Object timestampObj = map.get("timestamp");
        if (timestampObj instanceof String) {
            event.setTimestamp(LocalDateTime.parse((String) timestampObj));
        } else {
            event.setTimestamp(LocalDateTime.now());
        }
        
        return event;
    }
    
    return objectMapper.convertValue(eventObject, UserActionEvent.class);
}
```

**Benefits**:
- Accepts both Map and Object
- Flexible timestamp parsing
- Comprehensive error handling
- Rich debug logging

### Fix 2: Complete Post Deletion

#### PostEventConsumer.java

**Before** ❌:
```java
@KafkaListener(topics = "post_deleted")
public void handlePostDeleted(PostEvent event) {
    log.info("Received post_deleted event: {}", event.getPostId());
    
    postEmbeddingRepository.deleteByPostId(event.getPostId());
    redisCacheService.invalidateEmbedding(event.getPostId());
    redisCacheService.invalidateAllRecommendations();
    
    log.info("Successfully deleted post embedding: {}", event.getPostId());
}
```

**After** ✅:
```java
@KafkaListener(topics = "post_deleted")
public void handlePostDeleted(PostEvent event) {
    log.info("📥 Received post_deleted event: postId={}", event.getPostId());
    
    try {
        String postId = event.getPostId();
        
        // Delete from post_embeddings
        Optional<PostEmbedding> embeddingOpt = postEmbeddingRepository.findByPostId(postId);
        if (embeddingOpt.isPresent()) {
            postEmbeddingRepository.deleteByPostId(postId);
            log.info("✅ Deleted post embedding for: {}", postId);
        } else {
            log.warn("⚠️  Post embedding not found: {}", postId);
        }
        
        // Delete from user_feedback (NEW!)
        int deletedFeedback = userFeedbackRepository.deleteByPostId(postId);
        log.info("🗑️  Deleted {} user feedback records for post: {}", deletedFeedback, postId);
        
        // Invalidate caches
        redisCacheService.invalidateEmbedding(postId);
        redisCacheService.invalidateAllRecommendations();
        
        log.info("✅ Successfully processed post_deleted event for: {}", postId);
        
    } catch (Exception e) {
        log.error("❌ Error processing post_deleted event: {}", e.getMessage(), e);
    }
}
```

#### Added UserFeedbackRepository.deleteByPostId()

```java
@Transactional
@Modifying
@Query("DELETE FROM UserFeedback uf WHERE uf.postId = :postId")
int deleteByPostId(@Param("postId") String postId);
```

**Benefits**:
- Deletes ALL related data
- Checks existence before deletion
- Returns count of deleted records
- Comprehensive logging

## 📊 Data Flow

### User Interaction Flow

```
User clicks Like
    ↓
post-service records interaction
    ↓
post-service publishes to Kafka "user_action"
    ↓
{
  "actionType": "LIKE",
  "userId": "user123",
  "postId": "post456",
  "timestamp": "2024-12-09T12:00:00",
  "metadata": {"source": "post-service"}
}
    ↓
recommend-service consumes event
    ↓
Parse Map → UserActionEvent
    ↓
1. Save to user_feedback table
2. Update post_embeddings (likeCount++)
3. Recalculate popularity_score
4. Invalidate user recommendation cache
    ↓
✅ Done
```

### Post Deletion Flow

```
User deletes post
    ↓
post-service deletes from MongoDB
    ↓
post-service publishes to Kafka "post_deleted"
    ↓
recommend-service consumes event
    ↓
1. Delete from post_embeddings
2. Delete from user_feedback (all interactions)
3. Invalidate caches
    ↓
✅ Complete cleanup
```

## 🧪 Testing Guide

### Test User Interaction

```bash
TOKEN="your_jwt_token"

# 1. Create a post
POST_ID=$(curl -s -X POST http://localhost:8090/api/posts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "content": "Test post for interaction tracking"
  }' | jq -r '.id')

echo "Created post: $POST_ID"

# 2. Like the post
curl -X POST "http://localhost:8090/api/posts/$POST_ID/like" \
  -H "Authorization: Bearer $TOKEN"

# 3. Wait 2 seconds
sleep 2

# 4. Check recommend-service logs
docker-compose logs recommendation-service | grep "user_action"

# Expected:
# 📥 Received user_action: LIKE by user xxx on post yyy
# 💾 Saved user feedback: xxx -> yyy
# 📊 Updated engagement metrics for post yyy
# ✅ Successfully processed user_action event: LIKE

# 5. Check database
docker exec -it postgres psql -U postgres -d recommendation_db

# Query user_feedback
SELECT * FROM user_feedback WHERE post_id = '<POST_ID>' ORDER BY timestamp DESC;

# Query post_embeddings
SELECT post_id, like_count, comment_count, share_count, popularity_score 
FROM post_embeddings 
WHERE post_id = '<POST_ID>';
```

### Test Post Deletion

```bash
# 1. Get post with interactions
POST_ID="existing_post_id"

# 2. Check data before deletion
docker exec -it postgres psql -U postgres -d recommendation_db \
  -c "SELECT COUNT(*) FROM user_feedback WHERE post_id = '$POST_ID';"

docker exec -it postgres psql -U postgres -d recommendation_db \
  -c "SELECT * FROM post_embeddings WHERE post_id = '$POST_ID';"

# 3. Delete the post
curl -X DELETE "http://localhost:8090/api/posts/$POST_ID" \
  -H "Authorization: Bearer $TOKEN"

# 4. Wait 2 seconds
sleep 2

# 5. Check recommend-service logs
docker-compose logs recommendation-service | grep "post_deleted"

# Expected:
# 📥 Received post_deleted event: postId=xxx
# ✅ Deleted post embedding for: xxx
# 🗑️  Deleted 5 user feedback records for post: xxx
# ✅ Successfully processed post_deleted event for: xxx

# 6. Verify data deleted
docker exec -it postgres psql -U postgres -d recommendation_db \
  -c "SELECT COUNT(*) FROM user_feedback WHERE post_id = '$POST_ID';"
# Should return: 0

docker exec -it postgres psql -U postgres -d recommendation_db \
  -c "SELECT COUNT(*) FROM post_embeddings WHERE post_id = '$POST_ID';"
# Should return: 0
```

## 📋 Verification Checklist

### User Interactions
- [ ] Like action recorded in user_feedback
- [ ] post_embeddings.like_count incremented
- [ ] popularity_score recalculated
- [ ] Comment action recorded
- [ ] post_embeddings.comment_count incremented
- [ ] Share action recorded
- [ ] post_embeddings.share_count incremented
- [ ] User recommendation cache invalidated

### Post Deletion
- [ ] post_embeddings record deleted
- [ ] All user_feedback records deleted
- [ ] Embedding cache invalidated
- [ ] Recommendation cache invalidated
- [ ] No orphaned data in database

## 🔍 Debugging

### Check Kafka Topics

```bash
# List topics
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list

# Watch user_action topic
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic user_action \
  --from-beginning

# Watch post_deleted topic
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic post_deleted \
  --from-beginning
```

### Check Service Logs

```bash
# post-service
docker-compose logs -f post-service | grep -E "(user_action|interaction)"

# recommend-service
docker-compose logs -f recommendation-service | grep -E "(user_action|post_deleted)"
```

### Database Queries

```sql
-- Check recent user interactions
SELECT 
    uf.user_id,
    uf.post_id,
    uf.feedback_type,
    uf.feedback_value,
    uf.timestamp,
    pe.like_count,
    pe.comment_count,
    pe.popularity_score
FROM user_feedback uf
LEFT JOIN post_embeddings pe ON uf.post_id = pe.post_id
ORDER BY uf.timestamp DESC
LIMIT 20;

-- Check posts with no interactions
SELECT 
    post_id,
    like_count,
    comment_count,
    share_count,
    view_count,
    popularity_score
FROM post_embeddings
WHERE like_count = 0 AND comment_count = 0
ORDER BY created_at DESC
LIMIT 10;
```

## 🚨 Common Issues

### Issue: Still no user_feedback records
**Check**:
1. Kafka running? `docker-compose ps kafka`
2. post-service publishing? Check post-service logs
3. recommend-service consuming? Check recommend-service logs
4. Topic exists? `kafka-topics --list`

**Solution**:
```bash
# Restart services
docker-compose restart post-service recommendation-service

# Recreate Kafka topics
docker-compose down kafka
docker-compose up -d kafka
```

### Issue: Post deletion doesn't clean up
**Check**:
1. Is post_deleted event published?
2. Is recommend-service consuming it?
3. Check logs for errors

**Manual cleanup** (if needed):
```sql
DELETE FROM user_feedback WHERE post_id = 'xxx';
DELETE FROM post_embeddings WHERE post_id = 'xxx';
```

## 📁 Files Modified

### post-service
- `EventService.java` - Standardized event format ✅

### recommend-service
- `UserActionConsumer.java` - Flexible event parsing ✅
- `PostEventConsumer.java` - Complete deletion logic ✅
- `UserFeedbackRepository.java` - Added deleteByPostId() ✅

## 🎯 Impact

### Before
- ❌ User interactions not tracked
- ❌ Recommendations based on stale data
- ❌ Orphaned data after post deletion
- ❌ No debugging visibility

### After
- ✅ All interactions tracked in real-time
- ✅ Recommendations updated immediately
- ✅ Complete data cleanup on deletion
- ✅ Rich logging for debugging

---

**Fixed**: December 9, 2024  
**Status**: ✅ Complete  
**Impact**: User interactions now properly tracked, post deletion fully functional
