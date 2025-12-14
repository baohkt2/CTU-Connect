package vn.ctu.edu.recommend.model.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Response from Python friend ranking API
 * Python uses snake_case, so we need @JsonProperty annotations
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FriendRankingResponse {
    
    /**
     * List of ranked friend candidates
     */
    private List<RankedFriend> rankings;
    
    /**
     * Total count
     */
    private int count;
    
    /**
     * Model version used
     */
    @JsonProperty("model_version")
    private String modelVersion;
    
    /**
     * Ranked friend result from Python
     * All fields use @JsonProperty to map snake_case from Python to camelCase in Java
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RankedFriend {
        @JsonProperty("user_id")
        private String userId;
        
        @JsonProperty("final_score")
        private double finalScore;
        
        @JsonProperty("content_similarity")
        private double contentSimilarity;
        
        @JsonProperty("mutual_friends_score")
        private double mutualFriendsScore;
        
        @JsonProperty("academic_score")
        private double academicScore;
        
        @JsonProperty("activity_score")
        private double activityScore;
        
        @JsonProperty("recency_score")
        private double recencyScore;
        
        @JsonProperty("suggestion_type")
        private String suggestionType;
        
        @JsonProperty("suggestion_reason")
        private String suggestionReason;
    }
}
