package com.ctuconnect.event;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PasswordResetEvent {
    private Long userId;
    private String email;
    private String resetToken;
    private String eventType; // REQUESTED, COMPLETED
}
