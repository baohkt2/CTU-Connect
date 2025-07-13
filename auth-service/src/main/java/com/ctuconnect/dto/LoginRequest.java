package com.ctuconnect.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class LoginRequest {

    @NotBlank(message = "Email or username is required")
    private String identifier; // Có thể là email hoặc username

    @NotBlank(message = "Password is required")
    private String password;
}
