package com.ctuconnect.service;

import com.ctuconnect.client.UserServiceClient;
import com.ctuconnect.entity.NotificationEntity;
import com.ctuconnect.dto.NotificationDTO;
import com.ctuconnect.repository.NotificationRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationService {

    private final NotificationRepository notificationRepository;
    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final SimpMessagingTemplate messagingTemplate;
    private final RedisTemplate<String, Object> redisTemplate;
    private final UserServiceClient userServiceClient;

    private static final String NOTIFICATION_TOPIC = "notifications";
    private static final String UNREAD_COUNT_PREFIX = "unread_count:";

    /**
     * Create and send notification (Facebook-like notifications)
     */
    public void createNotification(String recipientId, String actorId, String type,
                                 String entityType, String entityId, String message) {

        NotificationEntity notification = NotificationEntity.builder()
                .recipientId(recipientId)
                .actorId(actorId)
                .type(NotificationEntity.NotificationType.valueOf(type))
                .entityType(entityType)
                .entityId(entityId)
                .message(message)
                .isRead(false)
                .createdAt(LocalDateTime.now())
                .build();

        // Save to database
        notification = notificationRepository.save(notification);

        // Update unread count in cache
        incrementUnreadCount(recipientId);

        // Send real-time notification via WebSocket
        sendRealTimeNotification(recipientId, notification);

        // Send to Kafka for further processing (email, push notifications, etc.)
        kafkaTemplate.send(NOTIFICATION_TOPIC, notification);

        log.info("Created notification for user {} from actor {}: {}", recipientId, actorId, message);
    }

    /**
     * Handle post interactions and create notifications
     */
    @KafkaListener(topics = "post-interactions")
    public void handlePostInteraction(PostInteractionEvent event) {
        String postAuthorId = event.getPostAuthorId();
        String actorId = event.getActorId();

        // Don't notify if user interacts with their own post
        if (postAuthorId.equals(actorId)) {
            return;
        }

        String message = "";
        String type = "";

        switch (event.getInteractionType()) {
            case "LIKE":
                message = event.getActorName() + " liked your post";
                type = "POST_LIKED";
                break;
            case "COMMENT":
                message = event.getActorName() + " commented on your post";
                type = "POST_COMMENTED";
                break;
            case "SHARE":
                message = event.getActorName() + " shared your post";
                type = "POST_SHARED";
                break;
        }

        createNotification(postAuthorId, actorId, type, "POST", event.getPostId(), message);
    }

    /**
     * Handle friend requests and friendship notifications
     */
    @KafkaListener(topics = "user-relationships")
    public void handleUserRelationship(UserRelationshipEvent event) {
        String message = "";
        String type = "";

        switch (event.getRelationshipType()) {
            case "FRIEND_REQUEST":
                message = event.getActorName() + " sent you a friend request";
                type = "FRIEND_REQUEST";
                createNotification(event.getTargetUserId(), event.getActorId(), type, "USER", event.getActorId(), message);
                break;
            case "FRIEND_ACCEPTED":
                message = event.getActorName() + " accepted your friend request";
                type = "FRIEND_ACCEPTED";
                createNotification(event.getTargetUserId(), event.getActorId(), type, "USER", event.getActorId(), message);
                break;
        }
    }

    /**
     * Bulk notification for post authors when their post gets popular
     */
    public void createBulkNotifications(Set<String> recipientIds, String actorId, String type,
                                      String entityType, String entityId, String message) {
        recipientIds.parallelStream().forEach(recipientId -> {
            if (!recipientId.equals(actorId)) { // Don't notify the actor
                createNotification(recipientId, actorId, type, entityType, entityId, message);
            }
        });
    }

    /**
     * Get user's notifications with pagination
     */
    public List<NotificationDTO> getUserNotifications(String userId, int page, int size) {
        List<NotificationEntity> notifications = notificationRepository
                .findByRecipientIdOrderByCreatedAtDesc(userId,
                        org.springframework.data.domain.PageRequest.of(page, size));

        return notifications.stream()
                .map(this::convertToDTO)
                .toList();
    }

    /**
     * Mark notification as read
     */
    public void markAsRead(String notificationId, String userId) {
        NotificationEntity notification = notificationRepository.findById(notificationId)
                .orElseThrow(() -> new RuntimeException("Notification not found"));

        if (!notification.getRecipientId().equals(userId)) {
            throw new RuntimeException("Unauthorized");
        }

        if (!notification.isRead()) {
            notification.setRead(true);
            notification.setReadAt(LocalDateTime.now());
            notificationRepository.save(notification);

            // Decrement unread count
            decrementUnreadCount(userId);
        }
    }

    /**
     * Mark all notifications as read for user
     */
    public void markAllAsRead(String userId) {
        List<NotificationEntity> unreadNotifications = notificationRepository
                .findByRecipientIdAndIsReadFalse(userId);

        unreadNotifications.forEach(notification -> {
            notification.setRead(true);
            notification.setReadAt(LocalDateTime.now());
        });

        notificationRepository.saveAll(unreadNotifications);

        // Reset unread count
        resetUnreadCount(userId);
    }

    /**
     * Get unread notification count
     */
    public long getUnreadCount(String userId) {
        String cacheKey = UNREAD_COUNT_PREFIX + userId;
        Object cachedCount = redisTemplate.opsForValue().get(cacheKey);

        if (cachedCount != null) {
            return ((Number) cachedCount).longValue();
        }

        // If not in cache, calculate from database
        long count = notificationRepository.countByRecipientIdAndIsReadFalse(userId);
        redisTemplate.opsForValue().set(cacheKey, count, 1, TimeUnit.HOURS);

        return count;
    }

    /**
     * Delete old notifications (cleanup job)
     */
    public void deleteOldNotifications() {
        LocalDateTime cutoffDate = LocalDateTime.now().minusDays(30);
        notificationRepository.deleteByCreatedAtBefore(cutoffDate);
    }

    private void sendRealTimeNotification(String userId, NotificationEntity notification) {
        try {
            NotificationDTO dto = convertToDTO(notification);
            messagingTemplate.convertAndSendToUser(userId, "/queue/notifications", dto);
        } catch (Exception e) {
            log.error("Failed to send real-time notification to user {}: {}", userId, e.getMessage());
        }
    }

    private void incrementUnreadCount(String userId) {
        String cacheKey = UNREAD_COUNT_PREFIX + userId;
        redisTemplate.opsForValue().increment(cacheKey);
        redisTemplate.expire(cacheKey, 1, TimeUnit.HOURS);
    }

    private void decrementUnreadCount(String userId) {
        String cacheKey = UNREAD_COUNT_PREFIX + userId;
        Long currentCount = redisTemplate.opsForValue().decrement(cacheKey);
        if (currentCount != null && currentCount < 0) {
            redisTemplate.opsForValue().set(cacheKey, 0);
        }
    }

    private void resetUnreadCount(String userId) {
        String cacheKey = UNREAD_COUNT_PREFIX + userId;
        redisTemplate.opsForValue().set(cacheKey, 0, 1, TimeUnit.HOURS);
    }

    private NotificationDTO convertToDTO(NotificationEntity notification) {
        return NotificationDTO.builder()
                .id(notification.getId())
                .recipientId(notification.getRecipientId())
                .actorId(notification.getActorId())
                .type(notification.getType().name())
                .entityType(notification.getEntityType())
                .entityId(notification.getEntityId())
                .message(notification.getMessage())
                .isRead(notification.isRead())
                .createdAt(notification.getCreatedAt())
                .readAt(notification.getReadAt())
                .build();
    }

    // Event classes for Kafka messaging
    @lombok.Data
    public static class PostInteractionEvent {
        private String postId;
        private String postAuthorId;
        private String actorId;
        private String actorName;
        private String interactionType; // LIKE, COMMENT, SHARE
    }

    @lombok.Data
    public static class UserRelationshipEvent {
        private String actorId;
        private String actorName;
        private String targetUserId;
        private String relationshipType; // FRIEND_REQUEST, FRIEND_ACCEPTED
    }
}
