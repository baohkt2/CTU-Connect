package vn.ctu.edu.recommend.kafka.producer;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.kafka.event.UserActionEvent;

import java.util.concurrent.CompletableFuture;

/**
 * Kafka producer for user interaction events
 * Sends events to Python training pipeline
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class UserInteractionProducer {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    private static final String POST_VIEWED_TOPIC = "post_viewed";
    private static final String POST_LIKED_TOPIC = "post_liked";
    private static final String POST_SHARED_TOPIC = "post_shared";
    private static final String POST_COMMENTED_TOPIC = "post_commented";
    private static final String USER_INTERACTION_TOPIC = "user_interaction";

    /**
     * Send post viewed event
     */
    public void sendPostViewedEvent(UserActionEvent event) {
        sendEvent(POST_VIEWED_TOPIC, event);
        sendEvent(USER_INTERACTION_TOPIC, event); // Also send to general topic
    }

    /**
     * Send post liked event
     */
    public void sendPostLikedEvent(UserActionEvent event) {
        sendEvent(POST_LIKED_TOPIC, event);
        sendEvent(USER_INTERACTION_TOPIC, event);
    }

    /**
     * Send post shared event
     */
    public void sendPostSharedEvent(UserActionEvent event) {
        sendEvent(POST_SHARED_TOPIC, event);
        sendEvent(USER_INTERACTION_TOPIC, event);
    }

    /**
     * Send post commented event
     */
    public void sendPostCommentedEvent(UserActionEvent event) {
        sendEvent(POST_COMMENTED_TOPIC, event);
        sendEvent(USER_INTERACTION_TOPIC, event);
    }

    /**
     * Generic method to send event to Kafka
     */
    private void sendEvent(String topic, UserActionEvent event) {
        try {
            CompletableFuture<SendResult<String, Object>> future = 
                kafkaTemplate.send(topic, event.getUserId(), event);

            future.whenComplete((result, ex) -> {
                if (ex != null) {
                    log.error("Failed to send event to topic {}: {}", topic, ex.getMessage());
                } else {
                    log.debug("Event sent to topic {}: user={}, post={}", 
                        topic, event.getUserId(), event.getPostId());
                }
            });
        } catch (Exception e) {
            log.error("Error sending event to topic {}: {}", topic, e.getMessage(), e);
        }
    }
}
