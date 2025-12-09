package vn.ctu.edu.recommend.kafka.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.kafka.event.UserActionEvent;
import vn.ctu.edu.recommend.model.entity.postgres.PostEmbedding;
import vn.ctu.edu.recommend.model.entity.postgres.UserFeedback;
import vn.ctu.edu.recommend.model.enums.FeedbackType;
import vn.ctu.edu.recommend.repository.postgres.PostEmbeddingRepository;
import vn.ctu.edu.recommend.repository.postgres.UserFeedbackRepository;
import vn.ctu.edu.recommend.repository.redis.RedisCacheService;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
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
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Handle user action events - supports both Map and UserActionEvent
     * Using @Payload to extract message payload directly
     */
    @KafkaListener(
        topics = "user_action", 
        groupId = "recommendation-service-group",
        containerFactory = "userActionKafkaListenerContainerFactory"
    )
    public void handleUserAction(@org.springframework.messaging.handler.annotation.Payload Object eventObject) {
        try {
            log.debug("📨 Raw event object type: {}", eventObject.getClass().getName());
            
            UserActionEvent event = parseUserActionEvent(eventObject);
            
            if (event == null || event.getActionType() == null || event.getUserId() == null || event.getPostId() == null) {
                log.warn("❌ Invalid user_action event: missing required fields");
                return;
            }
            
            log.info("📥 Received user_action event: {} by user {} on post {}", 
                event.getActionType(), event.getUserId(), event.getPostId());

            // Parse action type to feedback type
            FeedbackType feedbackType = parseFeedbackType(event.getActionType());
            if (feedbackType == null) {
                log.warn("⚠️  Unknown action type: {}", event.getActionType());
                return;
            }

            // Record feedback
            Float feedbackValue = getFeedbackValue(feedbackType);
            
            UserFeedback feedback = UserFeedback.builder()
                .userId(event.getUserId())
                .postId(event.getPostId())
                .feedbackType(feedbackType)
                .feedbackValue(feedbackValue)
                .context(event.getMetadata() != null ? event.getMetadata().toString() : null)
                .build();

            userFeedbackRepository.save(feedback);
            
            log.debug("💾 Saved user feedback: {} -> {}", event.getUserId(), event.getPostId());

            // Update post engagement metrics
            updateEngagementMetrics(event.getPostId(), feedbackType);

            // Invalidate user recommendation cache
            redisCacheService.invalidateRecommendations(event.getUserId());

            log.info("✅ Successfully processed user_action event: {}", feedbackType);

        } catch (Exception e) {
            log.error("❌ Error processing user_action event: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Parse event from various formats (Map or UserActionEvent object)
     */
    private UserActionEvent parseUserActionEvent(Object eventObject) {
        try {
            if (eventObject instanceof UserActionEvent) {
                return (UserActionEvent) eventObject;
            }
            
            if (eventObject instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> map = (Map<String, Object>) eventObject;
                
                UserActionEvent event = new UserActionEvent();
                event.setActionType(getStringValue(map, "actionType"));
                event.setUserId(getStringValue(map, "userId"));
                event.setPostId(getStringValue(map, "postId"));
                event.setMetadata(map.get("metadata"));
                
                // Parse timestamp - handle various formats
                Object timestampObj = map.get("timestamp");
                if (timestampObj instanceof String) {
                    try {
                        // Try ISO format first (from LocalDateTime.toString())
                        event.setTimestamp(LocalDateTime.parse((String) timestampObj));
                    } catch (Exception e) {
                        try {
                            // Try with DateTimeFormatter for different formats
                            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
                            event.setTimestamp(LocalDateTime.parse((String) timestampObj, formatter));
                        } catch (Exception e2) {
                            log.warn("⚠️ Failed to parse timestamp: {}, using current time", timestampObj);
                            event.setTimestamp(LocalDateTime.now());
                        }
                    }
                } else if (timestampObj instanceof Long) {
                    // Convert milliseconds since epoch to LocalDateTime
                    event.setTimestamp(LocalDateTime.now());
                } else {
                    event.setTimestamp(LocalDateTime.now());
                }
                
                log.debug("✅ Parsed Map to UserActionEvent: {}", event.getActionType());
                return event;
            }
            
            // Try to convert using ObjectMapper
            return objectMapper.convertValue(eventObject, UserActionEvent.class);
            
        } catch (Exception e) {
            log.error("❌ Error parsing user action event: {}", e.getMessage());
            return null;
        }
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
                
                log.debug("📊 Updated engagement metrics for post {}: likes={}, comments={}, shares={}",
                    postId, post.getLikeCount(), post.getCommentCount(), post.getShareCount());
            } else {
                log.warn("⚠️  Post embedding not found for postId: {}", postId);
            }
        } catch (Exception e) {
            log.error("❌ Error updating engagement metrics for post {}: {}", postId, e.getMessage());
        }
    }
}
