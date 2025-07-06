package com.ctuconnect.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class EmailVerificationDTO {

    @NotBlank(message = "Email is required")
    @Email(message = "Invalid email format")
    private String email;
    private String verificationCode;
    private String newPassword;
}