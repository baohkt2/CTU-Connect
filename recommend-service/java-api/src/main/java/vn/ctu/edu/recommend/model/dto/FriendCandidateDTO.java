package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * DTO for friend candidate from User Service
 * Must match FriendCandidateResponseDTO from user-service
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FriendCandidateDTO {
    
    private String userId;
    private String username;
    private String fullName;
    private String avatarUrl;
    private String bio;
    
    // Academic info
    private String facultyName;
    private String majorName;
    private Integer batchYear;  // Changed to Integer to match user-service
    
    // Social context
    private int mutualFriendsCount;
    private boolean sameFaculty;
    private boolean sameMajor;
    private boolean sameBatch;
    
    // Activity metrics
    private Double activityScore;
    
    // Skills and interests
    private List<String> skills;
    private List<String> interests;
    private List<String> courses;
    
    // Helper method to get batchYear as String for compatibility
    public String getBatchYearString() {
        return batchYear != null ? batchYear.toString() : null;
    }
}
