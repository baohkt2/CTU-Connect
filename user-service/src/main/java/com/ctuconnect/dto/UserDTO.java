package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.Set;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UserDTO {
    private String id;
    private String email;
    private String studentId;
    private String batch;
    private String fullName;
    private String role;
    private String college;
    private String faculty;
    private String major;
    private String gender;
    private String bio;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Set<String> friendIds;

    // For mutual friend and recommendation features
    private int mutualFriendsCount;
    private boolean sameCollege;
    private boolean sameFaculty;
    private boolean sameMajor;
}