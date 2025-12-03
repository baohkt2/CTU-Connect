package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.validation.constraints.Email;


@Data
@NoArgsConstructor
@AllArgsConstructor
public class AdminUpdateUserRequest {
    @Email(message = "Email should be valid")
    private String email;

    private String username;

    private String role;

    private Boolean isActive;

    private Boolean isEmailVerified;

    private String password; // For password reset by admin
}
