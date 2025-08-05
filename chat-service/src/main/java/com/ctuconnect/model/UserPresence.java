package com.ctuconnect.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.index.Indexed;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "user_presence")
public class UserPresence {
    @Id
    private String id;
    
    @Indexed(unique = true)
    private String userId;
    
    private String userName;
    
    private String userAvatar;
    
    private PresenceStatus status; // ONLINE, OFFLINE, AWAY
    
    private String currentActivity; // "typing in conversation_id" hoặc null
    
    private LocalDateTime lastSeenAt;
    
    private LocalDateTime updatedAt;
    
    private String sessionId; // WebSocket session ID
    
    public enum PresenceStatus {
        ONLINE,
        OFFLINE,
        AWAY
    }
}
