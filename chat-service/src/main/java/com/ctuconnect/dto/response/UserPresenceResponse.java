package com.ctuconnect.dto.response;

import com.ctuconnect.model.UserPresence;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserPresenceResponse {
    private String userId;
    private String userName;
    private String userAvatar;
    private UserPresence.PresenceStatus status;
    private String currentActivity;
    private LocalDateTime lastSeenAt;
    private String sessionId;
}
