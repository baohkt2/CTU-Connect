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
    private String username;
    private String fullName;
    private String avatarUrl;
    private String bio;
    
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
    
    public enum SuggestionType {
        MUTUAL_FRIENDS,
        ACADEMIC_CONNECTION,
        FRIENDS_OF_FRIENDS,
        PROFILE_VIEWER,
        SIMILAR_INTERESTS,
        LOCATION_BASED,
        ACTIVITY_BASED
    }
}
