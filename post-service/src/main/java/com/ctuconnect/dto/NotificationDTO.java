package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class NotificationDTO {
    private String id;
    private String recipientId;
    private String actorId;
    private String type;
    private String entityType;
    private String entityId;
    private String message;
    private boolean isRead;
    private LocalDateTime createdAt;
    private LocalDateTime readAt;
    private String imageUrl;
    private String actionUrl;
    
    // Actor information for display
    private String actorName;
    private String actorAvatarUrl;
}
