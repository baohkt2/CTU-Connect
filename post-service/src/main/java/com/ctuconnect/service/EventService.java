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

        kafkaTemplate.send("post-events", event);
    }

    public void publishInteractionEvent(String postId, String userId, String interactionType) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", "INTERACTION");
        event.put("postId", postId);
        event.put("userId", userId);
        event.put("interactionType", interactionType);
        event.put("timestamp", System.currentTimeMillis());

        kafkaTemplate.send("interaction-events", event);
    }

    public void publishCommentEvent(String eventType, String postId, String commentId, String authorId) {
        Map<String, Object> event = new HashMap<>();
        event.put("eventType", eventType);
        event.put("postId", postId);
        event.put("commentId", commentId);
        event.put("authorId", authorId);
        event.put("timestamp", System.currentTimeMillis());

        kafkaTemplate.send("comment-events", event);
    }
}
