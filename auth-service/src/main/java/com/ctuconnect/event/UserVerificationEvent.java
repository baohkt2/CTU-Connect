package com.ctuconnect.event;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserVerificationEvent {
    private Long userId;
    private String email;
    private String verificationToken;
    private boolean isVerified;
}
