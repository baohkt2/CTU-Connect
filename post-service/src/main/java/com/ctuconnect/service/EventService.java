package com.ctuconnect.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class EventService {

    @Autowired
    private KafkaTemplate<String, Object> kafkaTemplate;

    public void publishPostEvent(String eventType, String postId, String authorId, Object data) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", eventType);
        event.put("postId", postId);
        event.put("authorId", authorId);
        event.put("data", data);
        event.put("timestamp", System.currentTimeMillis());

        // Send to general post-events topic
        kafkaTemplate.send("post-events", event);
        
        // Send to specific topics for recommendation service
        if ("POST_CREATED".equals(eventType)) {
            kafkaTemplate.send("post_created", event);
        } else if ("POST_UPDATED".equals(eventType)) {
            kafkaTemplate.send("post_updated", event);
        } else if ("POST_DELETED".equals(eventType)) {
            kafkaTemplate.send("post_deleted", event);
        }
    }

    public void publishInteractionEvent(String postId, String userId, String interactionType) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "INTERACTION");
        event.put("postId", postId);
        event.put("userId", userId);
        event.put("interactionType", interactionType);
        event.put("actionType", interactionType); // Add actionType for recommendation service
        event.put("timestamp", System.currentTimeMillis());

        // Send to general interaction-events topic
        kafkaTemplate.send("interaction-events", event);
        
        // Send to user_action topic for recommendation service
        kafkaTemplate.send("user_action", event);
    }

    public void publishCommentEvent(String eventType, String postId, String commentId, String authorId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", eventType);
        event.put("postId", postId);
        event.put("commentId", commentId);
        event.put("authorId", authorId);
        event.put("timestamp", System.currentTimeMillis());

        kafkaTemplate.send("comment-events", event);
        
        // Send comment as user action for recommendation service
        Map<String, Object> userActionEvent = new HashMap<>();
        userActionEvent.put("postId", postId);
        userActionEvent.put("userId", authorId);
        userActionEvent.put("actionType", "COMMENT");
        userActionEvent.put("timestamp", System.currentTimeMillis());
        kafkaTemplate.send("user_action", userActionEvent);
    }
}
