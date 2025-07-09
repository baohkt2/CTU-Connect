package com.ctuconnect.event;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserRelationshipChangedEvent {
    private Long userId;
    private Long targetUserId;
    private String relationshipType; // FRIEND_REQUEST, FRIEND_ACCEPTED, FRIEND_REMOVED, BLOCKED, UNBLOCKED
    private String eventType; // CREATED, UPDATED, DELETED
}
