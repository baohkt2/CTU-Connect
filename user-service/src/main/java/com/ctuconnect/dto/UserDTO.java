package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.util.Set;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UserDTO {
    private String id;
    private String email;
    private String username; // Added missing username field
    private String studentId;
    private Integer batch; // Changed to Integer to match Neo4j dataset
    private String fullName;
    private String role;
    private String college;
    private String faculty;
    private String major;
    private String gender;
    private String bio;
    private Boolean isActive; // Added missing isActive field
    private OffsetDateTime createdAt;
    private OffsetDateTime updatedAt;
    private Set<String> friendIds; // Changed from Long to String for UUID consistency

    // For mutual friend and recommendation features
    private int mutualFriendsCount;
    private boolean sameCollege;
    private boolean sameFaculty;
    private boolean sameMajor;
}