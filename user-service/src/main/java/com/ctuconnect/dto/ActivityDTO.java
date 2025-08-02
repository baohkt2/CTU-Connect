package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ActivityDTO {
    private String id;
    private String userId;
    private String activityType; // POST_CREATED, POST_LIKED, COMMENT_ADDED, etc.
    private String targetType; // POST, COMMENT, USER
    private String targetId;
    private String entityType; // Add missing entityType field
    private String entityId; // Add missing entityId field
    private String description;
    private LocalDateTime timestamp;
    private String actorName;
    private String actorAvatarUrl;

    // For activity feed display
    private String displayText;
    private String actionUrl;
    private boolean isRead;

    public enum ActivityType {
        POST_CREATED,
        POST_LIKED,
        POST_SHARED,
        COMMENT_ADDED,
        FRIEND_REQUEST_SENT,
        FRIEND_REQUEST_ACCEPTED,
        PROFILE_UPDATED
    }

    public enum TargetType {
        POST,
        COMMENT,
        USER,
        PROFILE
    }

    public enum EntityType {
        POST,
        COMMENT,
        USER,
        PROFILE
    }
}
