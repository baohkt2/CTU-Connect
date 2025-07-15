package com.ctuconnect.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class ResendVerificationRequest {
    @NotBlank(message = "Token is required")
    private String token;
}
