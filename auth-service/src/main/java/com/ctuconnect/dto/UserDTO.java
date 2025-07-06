package com.ctuconnect.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class UserDTO {

    private Long id;

    @NotBlank(message = "Email is required")
    private String email;

    @NotBlank(message = "Full name is required")
    @Size(min = 2, max = 100, message = "Full name must be between 2 and 100 characters")
    private String fullName;

    @Size(max = 50, message = "Username must be less than 50 characters")
    private String username;

    @Size(max = 500, message = "Bio must be less than 500 characters")
    private String bio;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private boolean isVerified;
}