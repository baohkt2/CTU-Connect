package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class FriendSuggestionDTO {
    private String userId;
    private String id;  // Alias for userId - for frontend compatibility
    private String username;
    private String fullName;
    private String avatarUrl;
    private String bio;
    private String email;
    private String studentId;
    
    // Suggestion metadata
    private int mutualFriendsCount;
    private String suggestionReason;
    private double relevanceScore;
    private SuggestionType suggestionType;
    
    // Academic context
    private String facultyName;
    private String majorName;
    private String batchYear;
    
    // Social context
    private boolean sameCollege;
    private boolean sameFaculty;
    private boolean sameMajor;
    private boolean sameBatch;
    
    // ML-enhanced scores (from recommend-service)
    private double contentSimilarity;
    private double academicScore;
    private double activityScore;
    private double mutualFriendsScore;
    
    // Flag to indicate if this suggestion came from ML service
    @Builder.Default
    private boolean mlEnhanced = false;
    
    public enum SuggestionType {
        MUTUAL_FRIENDS,
        ACADEMIC_CONNECTION,
        FRIENDS_OF_FRIENDS,
        PROFILE_VIEWERS,  // Renamed for consistency
        PROFILE_VIEWER,   // Backward compatibility
        SIMILAR_INTERESTS,
        CONTENT_SIMILARITY,  // New ML-based type
        LOCATION_BASED,
        ACTIVITY_BASED
    }
}
