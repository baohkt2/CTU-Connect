package vn.ctu.edu.recommend.kafka.consumer;

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

    @KafkaListener(topics = "user_action", groupId = "recommendation-service-group")
    public void handleUserAction(UserActionEvent event) {
        log.info("Received user_action event: {} on post {}", 
            event.getActionType(), event.getPostId());

        try {
            // Parse action type to feedback type
            FeedbackType feedbackType = parseFeedbackType(event.getActionType());
            if (feedbackType == null) {
                log.warn("Unknown action type: {}", event.getActionType());
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

            // Update post engagement metrics
            updateEngagementMetrics(event.getPostId(), feedbackType);

            // Invalidate user recommendation cache
            redisCacheService.invalidateRecommendations(event.getUserId());

            log.info("Successfully processed user_action event");

        } catch (Exception e) {
            log.error("Error processing user_action event: {}", e.getMessage(), e);
        }
    }

    private FeedbackType parseFeedbackType(String actionType) {
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
        }
    }
}
