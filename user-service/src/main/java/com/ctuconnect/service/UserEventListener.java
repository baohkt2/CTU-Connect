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
                    .authId(Long.valueOf(event.get("userId").toString()))
                    .email(event.get("email").toString())
                    .username(event.get("username").toString())
                    .role(event.get("role").toString())
                    .isActive(true)
                    .build();

            userRepository.save(user);

            log.info("User created successfully in Neo4j with authId: {}", user.getAuthId());
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

            Long authId = Long.valueOf(event.get("userId").toString());
            UserEntity user = userRepository.findByAuthId(authId)
                    .orElseThrow(() -> new RuntimeException("User not found with authId: " + authId));

            user.setEmail(event.get("email").toString());
            user.setUsername(event.get("username").toString());
            user.setRole(event.get("role").toString());
            user.setIsActive(Boolean.valueOf(event.get("isActive").toString()));

            userRepository.save(user);

            log.info("User updated successfully in Neo4j with authId: {}", authId);
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

            Long authId = Long.valueOf(event.get("userId").toString());
            UserEntity user = userRepository.findByAuthId(authId)
                    .orElseThrow(() -> new RuntimeException("User not found with authId: " + authId));

            userRepository.delete(user);

            log.info("User deleted successfully from Neo4j with authId: {}", authId);
            acknowledgment.acknowledge();

        } catch (Exception e) {
            log.error("Error processing user deleted event: {}", e.getMessage(), e);
            // Don't acknowledge - message will be retried
        }
    }
}
