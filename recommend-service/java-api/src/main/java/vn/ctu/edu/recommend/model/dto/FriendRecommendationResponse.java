package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Response object for friend recommendations API
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FriendRecommendationResponse {
    
    /**
     * User ID for whom recommendations were generated
     */
    private String userId;
    
    /**
     * List of friend suggestions
     */
    private List<FriendSuggestion> suggestions;
    
    /**
     * Total count of suggestions
     */
    private int count;
    
    /**
     * Response metadata
     */
    private ResponseMetadata metadata;
    
    /**
     * Individual friend suggestion
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class FriendSuggestion {
        // User info
        private String userId;
        private String username;
        private String fullName;
        private String avatarUrl;
        private String bio;
        
        // Academic context
        private String facultyName;
        private String majorName;
        private String batchYear;
        
        // Social context flags
        private boolean sameFaculty;
        private boolean sameMajor;
        private boolean sameBatch;
        private int mutualFriendsCount;
        
        // Scoring details
        private double relevanceScore;
        private double contentSimilarity;
        private double mutualFriendsScore;
        private double academicScore;
        private double activityScore;
        
        // Suggestion metadata
        private String suggestionType;
        private String suggestionReason;
        private int rankPosition;
    }
    
    /**
     * Response metadata
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ResponseMetadata {
        private String source; // "ml", "cache", "fallback"
        private long processingTimeMs;
        private LocalDateTime timestamp;
        private String modelVersion;
        private boolean mlEnabled;
    }
    
    /**
     * Builder helper for empty response
     */
    public static FriendRecommendationResponse empty(String userId) {
        return FriendRecommendationResponse.builder()
            .userId(userId)
            .suggestions(List.of())
            .count(0)
            .metadata(ResponseMetadata.builder()
                .source("empty")
                .processingTimeMs(0)
                .timestamp(LocalDateTime.now())
                .build())
            .build();
    }
}
