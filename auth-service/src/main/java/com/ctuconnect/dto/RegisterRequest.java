package com.ctuconnect.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class RegisterRequest {

    @NotBlank(message = "Email is required")
    @Email(message = "Email must be valid")
    @Pattern(
            regexp = "^[A-Za-z0-9._%+-]+@(?:student\\.)?ctu\\.edu\\.vn$",
            message = "Email must end with @ctu.edu.vn or @student.ctu.edu.vn"
    )
    private String email;

    @NotBlank(message = "Username is required")
    @Pattern(
            regexp = "^[a-zA-Z][a-zA-Z0-9._]{2,24}$",
            message = "Username must be 3–25 characters, start with a letter, and only contain letters, numbers, dots or underscores"
    )
    private String username;

    @NotBlank(message = "Password is required")
    @Pattern(
            regexp = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=!])(?=\\S+$).{8,20}$",
            message = "Password must be 8–20 characters long, include uppercase, lowercase, number, and special character, with no spaces"
    )
    private String password;


    private String role;
}
