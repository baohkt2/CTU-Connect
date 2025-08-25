package com.ctuconnect.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import com.fasterxml.jackson.annotation.JsonInclude;


import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class UserProfileDTO {
    private String id;

    @Email(message = "Email format is invalid")
    @NotBlank(message = "Email is required")
    private String email;

    @Size(min = 3, max = 50, message = "Username must be between 3 and 50 characters")
    private String username;

    @Size(max = 20, message = "Student ID must not exceed 20 characters")
    private String studentId;

    @Size(max = 100, message = "Full name must not exceed 100 characters")
    private String fullName;

    @Size(max = 500, message = "Bio must not exceed 500 characters")
    private String bio;

    private String role;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Academic information
    private String college;
    private String faculty;
    private String major;
    private Integer batch;
    private String gender;

    // Social information
    private Long friendsCount;
    private Long mutualFriendsCount;
    private Long sentRequestsCount;
    private Long receivedRequestsCount;

    // Relationship flags
    private Boolean isFriend;
    private Boolean requestSent;
    private Boolean requestReceived;
    private Boolean sameCollege;
    private Boolean sameFaculty;
    private Boolean sameMajor;
    private Boolean sameBatch;
}
