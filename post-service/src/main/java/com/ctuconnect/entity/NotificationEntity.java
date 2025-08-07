package com.ctuconnect.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Document(collection = "notifications")
public class NotificationEntity {
    
    @Id
    private String id;
    
    private String recipientId;
    private String actorId;
    
    private NotificationType type;
    private String entityType; // POST, COMMENT, USER, etc.
    private String entityId;
    
    private String message;
    private boolean isRead;
    
    @Field("created_at")
    @CreatedDate
    private LocalDateTime createdAt;
    
    private LocalDateTime readAt;
    
    // Additional metadata for rich notifications
    private String imageUrl;
    private String actionUrl;
    
    public enum NotificationType {
        POST_LIKED,
        POST_COMMENTED,
        POST_SHARED,
        COMMENT_REPLIED,
        FRIEND_REQUEST,
        FRIEND_ACCEPTED,
        MENTION,
        TAG,
        BIRTHDAY,
        EVENT_REMINDER,
        GROUP_INVITATION,
        SYSTEM_NOTIFICATION
    }
}
