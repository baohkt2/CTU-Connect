package com.ctuconnect.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class PasswordResetRequest {

    @Email(message = "Email must be valid")
    private String email;

    private String token;

    @Size(min = 8, message = "Password must be at least 8 characters")
    private String newPassword;
}
