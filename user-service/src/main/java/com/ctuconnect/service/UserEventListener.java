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

import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserEventListener {

    private final UserRepository userRepository;

    @KafkaListener(topics = "user-registration", groupId = "user-service-group")
    @Transactional
    public void handleUserCreatedEvent(@Payload Map<String, Object> event,
                                     @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                     @Header(KafkaHeaders.RECEIVED_PARTITION) int partition,
                                     @Header(KafkaHeaders.OFFSET) long offset,
                                     Acknowledgment acknowledgment) {
        try {
            log.info("Received user created event: {}", event);

            // Create user in Neo4j
            UserEntity user = UserEntity.builder()
                    .id(String.valueOf(event.get("userId"))) // Ensure String conversion
                    .email(String.valueOf(event.get("email")))
                    .username(String.valueOf(event.get("username")))
                    .role(String.valueOf(event.get("role")))
                    .isActive(true)
                    .build();

            userRepository.save(user);

            log.info("User created successfully in Neo4j with id: {}", user.getId());
            acknowledgment.acknowledge();

        } catch (Exception e) {
            log.error("Error processing user created event: {}", e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }

    @KafkaListener(topics = "user-updated", groupId = "user-service-group")
    @Transactional
    public void handleUserUpdatedEvent(@Payload Map<String, Object> event,
                                     @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                     Acknowledgment acknowledgment) {
        try {
            log.info("Received user updated event: {}", event);

            String userId = String.valueOf(event.get("userId")); // Ensure String conversion
            UserEntity user = userRepository.findById(userId)
                    .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

            user.setEmail(String.valueOf(event.get("email")));
            user.setUsername(String.valueOf(event.get("username")));
            user.setRole(String.valueOf(event.get("role")));
            user.setIsActive(Boolean.valueOf(String.valueOf(event.get("isActive"))));

            userRepository.save(user);

            log.info("User updated successfully in Neo4j with id: {}", userId);
            acknowledgment.acknowledge();

        } catch (Exception e) {
            log.error("Error processing user updated event: {}", e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }

    @KafkaListener(topics = "user-deleted", groupId = "user-service-group")
    @Transactional
    public void handleUserDeletedEvent(@Payload Map<String, Object> event,
                                     @Header(KafkaHeaders.RECEIVED_TOPIC) String topic,
                                     Acknowledgment acknowledgment) {
        try {
            log.info("Received user deleted event: {}", event);

            String userId = event.get("userId").toString();
            UserEntity user = userRepository.findById(userId)
                    .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

            userRepository.delete(user);

            log.info("User deleted successfully from Neo4j with id: {}", userId);
            acknowledgment.acknowledge();

        } catch (Exception e) {
            log.error("Error processing user deleted event: {}", e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }
}
