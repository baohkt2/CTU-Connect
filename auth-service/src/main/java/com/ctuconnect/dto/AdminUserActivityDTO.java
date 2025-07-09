package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AdminUserActivityDTO {
    private Long userId;
    private String email;
    private String username;
    private String activity; // LOGIN, LOGOUT, PASSWORD_RESET, etc.
    private String ipAddress;
    private String userAgent;
    private LocalDateTime timestamp;
    private boolean success;
    private String errorMessage;
}
