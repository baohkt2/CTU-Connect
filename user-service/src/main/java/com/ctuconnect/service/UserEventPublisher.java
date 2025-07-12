package com.ctuconnect.service;

import com.ctuconnect.event.UserProfileUpdatedEvent;
import com.ctuconnect.event.UserRelationshipChangedEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserEventPublisher {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    public void publishUserCreatedEvent(String userId, String email, String username, String role) {
        try {
            Map<String, Object> event = new HashMap<>();
            event.put("userId", userId);
            event.put("email", email);
            event.put("username", username);
            event.put("role", role);

            kafkaTemplate.send("user-created", userId, event);
            log.info("Published user created event for user: {}", userId);

        } catch (Exception e) {
            log.error("Failed to publish user created event for user: {}", userId, e);
        }
    }

    public void publishUserDeletedEvent(String userId, String email) {
        try {
            Map<String, Object> event = new HashMap<>();
            event.put("userId", userId);
            event.put("email", email);

            kafkaTemplate.send("user-deleted", userId, event);
            log.info("Published user deleted event for user: {}", userId);

        } catch (Exception e) {
            log.error("Failed to publish user deleted event for user: {}", userId, e);
        }
    }

    public void publishUserProfileUpdatedEvent(String userId, String email, String username,
                                             String firstName, String lastName, String bio, String profilePicture) {
        try {
            Map<String, Object> event = new HashMap<>();
            event.put("userId", userId);
            event.put("email", email);
            event.put("username", username);
            event.put("firstName", firstName);
            event.put("lastName", lastName);
            event.put("bio", bio);
            event.put("profilePicture", profilePicture);

            kafkaTemplate.send("user-profile-updated", userId, event);
            log.info("Published user profile updated event for user: {}", userId);

        } catch (Exception e) {
            log.error("Failed to publish user profile updated event for user: {}", userId, e);
        }
    }

    public void publishUserRelationshipChangedEvent(String userId, String targetUserId,
                                                  String relationshipType, String eventType) {
        try {
            Map<String, Object> event = new HashMap<>();
            event.put("userId", userId);
            event.put("targetUserId", targetUserId);
            event.put("relationshipType", relationshipType);
            event.put("eventType", eventType);

            kafkaTemplate.send("user-relationship-changed", userId, event);
            log.info("Published user relationship changed event: user {} -> {}, type: {}",
                    userId, targetUserId, relationshipType);

        } catch (Exception e) {
            log.error("Failed to publish user relationship changed event for user: {}", userId, e);
        }
    }
}
