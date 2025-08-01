package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.Map;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ActivityDTO {
    private String id;
    private String userId;
    private String activityType; // POST_CREATED, POST_LIKED, COMMENT_ADDED, FRIEND_ADDED, etc.
    private String entityType; // POST, COMMENT, USER
    private String entityId;
    private String description;
    private Map<String, Object> metadata;
    private LocalDateTime timestamp;

    // Display information
    private String actorName;
    private String actorAvatarUrl;
    private String targetName;
    private String previewText;
    private String actionUrl;

    public enum ActivityType {
        POST_CREATED,
        POST_LIKED,
        POST_COMMENTED,
        POST_SHARED,
        FRIEND_ADDED,
        PROFILE_UPDATED,
        GROUP_JOINED,
        EVENT_ATTENDED
    }
}
