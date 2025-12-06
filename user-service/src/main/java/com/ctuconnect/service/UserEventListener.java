package com.ctuconnect.service;

import com.ctuconnect.event.UserUpdatedEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.handler.annotation.Header;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Service;

import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserEventListener {

    private final UserSyncService userSyncService;

    @KafkaListener(topics = "user-registration", groupId = "user-service-group")
    public void handleUserCreatedEvent(@Payload Map<String, Object> event,
                                       @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                       @Header(KafkaHeaders.OFFSET) long offset,
                                       Acknowledgment acknowledgment) {
        try {
            log.info("Received user-created event from topic '{}' at offset {}: {}", topic, offset, event);

            String userId = String.valueOf(event.get("userId"));
            String email = String.valueOf(event.get("email"));
            String username = String.valueOf(event.get("username"));
            String role = String.valueOf(event.get("role"));

            userSyncService.createUserFromAuthService(userId, email, username, role);

            log.info("User created successfully in user-db: {}", userId);
            acknowledgment.acknowledge();
        } catch (Exception e) {
            log.error("Failed to process user-created event at offset {}: {}", offset, e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }

    @KafkaListener(topics = "user-updated", groupId = "user-service-group")
    public void handleUserUpdatedEvent(@Payload UserUpdatedEvent event,
                                       @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                       Acknowledgment acknowledgment) {
        try {
            log.info("Received user-updated event from topic '{}': {}", topic, event);

            String userId = event.getUserId();
            String email = event.getEmail();
            String role = event.getRole();

            userSyncService.updateUserFromAuth(userId, email, role);

            log.info("User updated successfully: {}", userId);
            acknowledgment.acknowledge();
        } catch (Exception e) {
            log.error("Failed to process user-updated event: {}", e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }

    @KafkaListener(topics = "user-deleted", groupId = "user-service-group")
    public void handleUserDeletedEvent(@Payload Map<String, Object> event,
                                       @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                       Acknowledgment acknowledgment) {
        try {
            log.info("Received user-deleted event from topic '{}': {}", topic, event);

            String userId = String.valueOf(event.get("userId"));
            userSyncService.deleteUserFromAuth(userId);

            log.info("User deleted successfully: {}", userId);
            acknowledgment.acknowledge();
        } catch (Exception e) {
            log.error("Failed to process user-deleted event: {}", e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }
}
