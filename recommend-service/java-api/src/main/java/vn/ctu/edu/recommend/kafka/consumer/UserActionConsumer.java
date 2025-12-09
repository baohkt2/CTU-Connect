package vn.ctu.edu.recommend.kafka.consumer;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.model.entity.postgres.PostEmbedding;
import vn.ctu.edu.recommend.model.entity.postgres.UserFeedback;
import vn.ctu.edu.recommend.model.enums.FeedbackType;
import vn.ctu.edu.recommend.repository.postgres.PostEmbeddingRepository;
import vn.ctu.edu.recommend.repository.postgres.UserFeedbackRepository;
import vn.ctu.edu.recommend.repository.redis.RedisCacheService;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Map;
import java.util.Optional;

/**
 * Kafka consumer for user action events
 * Updates engagement metrics and user feedback
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class UserActionConsumer {

    private final UserFeedbackRepository userFeedbackRepository;
    private final PostEmbeddingRepository postEmbeddingRepository;
    private final RedisCacheService redisCacheService;

    /**
     * Handle user action events from Kafka - receives Map directly
     */
    @KafkaListener(
        topics = "user_action", 
        groupId = "recommendation-service-group",
        containerFactory = "userActionKafkaListenerContainerFactory"
    )
    public void handleUserAction(@Payload Map<String, Object> eventMap) {
        try {
            log.debug("📨 Received user_action event map with keys: {}", eventMap.keySet());
            
            // Extract fields from map
            String actionType = getStringValue(eventMap, "actionType");
            String userId = getStringValue(eventMap, "userId");
            String postId = getStringValue(eventMap, "postId");
            Object metadata = eventMap.get("metadata");
            LocalDateTime timestamp = parseTimestamp(eventMap.get("timestamp"));
            
            // Validate required fields
            if (actionType == null || userId == null || postId == null) {
                log.warn("❌ Invalid user_action event: missing required fields - actionType: {}, userId: {}, postId: {}", 
                    actionType, userId, postId);
                return;
            }
            
            log.info("📥 Received user_action: {} by user {} on post {}", 
                actionType, userId, postId);

            // Parse action type to feedback type
            FeedbackType feedbackType = parseFeedbackType(actionType);
            if (feedbackType == null) {
                log.warn("⚠️  Unknown action type: {}", actionType);
                return;
            }

            // Record feedback
            Float feedbackValue = getFeedbackValue(feedbackType);
            
            UserFeedback feedback = UserFeedback.builder()
                .userId(userId)
                .postId(postId)
                .feedbackType(feedbackType)
                .feedbackValue(feedbackValue)
                .context(metadata != null ? metadata.toString() : null)
                .build();

            userFeedbackRepository.save(feedback);
            
            log.debug("💾 Saved user feedback: {} -> {} (type: {}, value: {})", 
                userId, postId, feedbackType, feedbackValue);

            // Update post engagement metrics
            updateEngagementMetrics(postId, feedbackType);

            // Invalidate user recommendation cache
            redisCacheService.invalidateRecommendations(userId);
            log.debug("🗑️  Invalidated cache for user: {}", userId);

            log.info("✅ Successfully processed user_action event: {} (feedback: {})", 
                actionType, feedbackType);

        } catch (Exception e) {
            log.error("❌ Error processing user_action event: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Parse timestamp from various formats
     */
    private LocalDateTime parseTimestamp(Object timestampObj) {
        if (timestampObj == null) {
            return LocalDateTime.now();
        }
        
        if (timestampObj instanceof Long) {
            return LocalDateTime.now();
        }
        
        if (timestampObj instanceof String) {
            String timestampStr = (String) timestampObj;
            try {
                // Try ISO format first (LocalDateTime.toString() format)
                return LocalDateTime.parse(timestampStr);
            } catch (DateTimeParseException e) {
                try {
                    // Try with custom formatter for microseconds
                    DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
                    return LocalDateTime.parse(timestampStr, formatter);
                } catch (DateTimeParseException e2) {
                    log.warn("⚠️ Failed to parse timestamp: {}, using current time", timestampStr);
                    return LocalDateTime.now();
                }
            }
        }
        
        return LocalDateTime.now();
    }
    
    private String getStringValue(Map<String, Object> map, String key) {
        Object value = map.get(key);
        return value != null ? value.toString() : null;
    }

    private FeedbackType parseFeedbackType(String actionType) {
        if (actionType == null) return null;
        
        return switch (actionType.toUpperCase()) {
            case "LIKE", "LIKED" -> FeedbackType.LIKE;
            case "COMMENT", "COMMENTED" -> FeedbackType.COMMENT;
            case "SHARE", "SHARED" -> FeedbackType.SHARE;
            case "SAVE", "SAVED" -> FeedbackType.SAVE;
            case "VIEW", "VIEWED" -> FeedbackType.VIEW;
            case "CLICK", "CLICKED" -> FeedbackType.CLICK;
            case "SKIP", "SKIPPED" -> FeedbackType.SKIP;
            case "HIDE", "HIDDEN" -> FeedbackType.HIDE;
            case "REPORT", "REPORTED" -> FeedbackType.REPORT;
            default -> null;
        };
    }

    private Float getFeedbackValue(FeedbackType feedbackType) {
        return switch (feedbackType) {
            case LIKE -> 1.0f;
            case COMMENT -> 2.0f;
            case SHARE -> 3.0f;
            case SAVE -> 2.5f;
            case VIEW -> 0.5f;
            case CLICK -> 0.8f;
            case SKIP -> -0.5f;
            case HIDE -> -2.0f;
            case REPORT -> -3.0f;
            case DWELL_TIME -> 1.0f;
        };
    }

    private void updateEngagementMetrics(String postId, FeedbackType feedbackType) {
        try {
            Optional<PostEmbedding> postOpt = postEmbeddingRepository.findByPostId(postId);
            if (postOpt.isPresent()) {
                PostEmbedding post = postOpt.get();
                
                switch (feedbackType) {
                    case LIKE -> post.setLikeCount(post.getLikeCount() + 1);
                    case COMMENT -> post.setCommentCount(post.getCommentCount() + 1);
                    case SHARE -> post.setShareCount(post.getShareCount() + 1);
                    case VIEW -> post.setViewCount(post.getViewCount() + 1);
                }
                
                post.calculatePopularityScore();
                postEmbeddingRepository.save(post);
                
                log.info("📊 Updated engagement for post {}: likes={}, comments={}, shares={}, views={}, popularity={}",
                    postId, post.getLikeCount(), post.getCommentCount(), 
                    post.getShareCount(), post.getViewCount(), post.getPopularityScore());
            } else {
                log.warn("⚠️  Post embedding not found for postId: {}", postId);
            }
        } catch (Exception e) {
            log.error("❌ Error updating engagement metrics for post {}: {}", postId, e.getMessage());
        }
    }
}
