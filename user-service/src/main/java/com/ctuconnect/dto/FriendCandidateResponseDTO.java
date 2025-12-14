package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

/**
 * Friend candidate DTO for recommend-service
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FriendCandidateResponseDTO implements Serializable {
    private String userId;
    private String username;
    private String fullName;
    private String avatarUrl;
    private String bio;
    
    // Academic info
    private String facultyName;
    private String majorName;
    private Integer batchYear;
    
    // Social context
    private boolean sameFaculty;
    private boolean sameMajor;
    private boolean sameBatch;
    private int mutualFriendsCount;
    
    // Activity metrics
    private Double activityScore;
    
    // Optional fields
    private List<String> skills;
    private List<String> interests;
    private List<String> courses;
}
