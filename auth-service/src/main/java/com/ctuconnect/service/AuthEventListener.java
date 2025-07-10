package com.ctuconnect.service;

import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.handler.annotation.Header;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthEventListener {

    private final UserRepository userRepository;

    @KafkaListener(topics = "user-profile-updated", groupId = "auth-service-group")
    @Transactional
    public void handleUserProfileUpdatedEvent(@Payload Map<String, Object> event,
                                            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                            Acknowledgment acknowledgment) {
        try {
            log.info("Received user profile updated event: {}", event);

            String userId = event.get("userId").toString();
            UserEntity user = userRepository.findById(userId)
                    .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

            // Update user profile information
            user.setEmail(event.get("email").toString());
            user.setUsername(event.get("username").toString());
            user.setUpdatedAt(LocalDateTime.now());

            userRepository.save(user);

            log.info("User profile updated successfully in auth service for user: {}", userId);
            acknowledgment.acknowledge();

        } catch (Exception e) {
            log.error("Error processing user profile updated event: {}", e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }

    @KafkaListener(topics = "user-relationship-changed", groupId = "auth-service-group")
    @Transactional
    public void handleUserRelationshipChangedEvent(@Payload Map<String, Object> event,
                                                  @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                                  Acknowledgment acknowledgment) {
        try {
            log.info("Received user relationship changed event: {}", event);

            // This could be used for analytics, notifications, etc.
            // For now, we'll just log it
            String userId = event.get("userId").toString();
            String targetUserId = event.get("targetUserId").toString();
            String relationshipType = event.get("relationshipType").toString();
            String eventType = event.get("eventType").toString();

            log.info("User relationship changed: {} -> {}, type: {}, event: {}",
                    userId, targetUserId, relationshipType, eventType);

            acknowledgment.acknowledge();

        } catch (Exception e) {
            log.error("Error processing user relationship changed event: {}", e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }
}
