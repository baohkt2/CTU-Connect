package vn.ctu.edu.recommend.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * Request object for Python friend ranking API
 * Uses @JsonProperty for snake_case serialization to match Python expectations
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FriendRankingRequest {
    
    /**
     * Current user profile
     */
    @JsonProperty("current_user")
    private UserProfileData currentUser;
    
    /**
     * List of candidate users
     */
    private List<UserProfileData> candidates;
    
    /**
     * Additional scores per candidate (keyed by user_id)
     */
    @JsonProperty("additional_scores")
    private Map<String, AdditionalScores> additionalScores;
    
    /**
     * Number of top candidates to return
     */
    @JsonProperty("top_k")
    @Builder.Default
    private int topK = 20;
    
    /**
     * User profile data for embedding generation
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class UserProfileData {
        @JsonProperty("user_id")
        private String userId;
        private String major;
        private String faculty;
        private List<String> courses;
        private List<String> skills;
        private String bio;
    }
    
    /**
     * Additional scores from Java side
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class AdditionalScores {
        @JsonProperty("mutual_friends_score")
        private double mutualFriendsScore;
        
        @JsonProperty("academic_score")
        private double academicScore;
        
        @JsonProperty("activity_score")
        private double activityScore;
        
        @JsonProperty("recency_score")
        private double recencyScore;
        
        @JsonProperty("mutual_friends_count")
        private int mutualFriendsCount;
    }
}
