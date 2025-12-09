package com.ctuconnect.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Response DTO for recommendation feed from recommendation-service
 * Mirrors the structure from recommend-service RecommendationResponse
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class RecommendationFeedResponse {
    
    private String userId;
    private List<RecommendedPost> recommendations;
    private Integer totalCount;
    private Integer page;
    private Integer size;
    private String abVariant;
    private LocalDateTime timestamp;
    private Long processingTimeMs;

    /**
     * Recommended post with score information
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class RecommendedPost {
        private String postId;
        private String authorId;
        private String content;
        private Double score;
        private Float finalScore;
        private Float contentSimilarity;
        private Float graphRelationScore;
        private Float academicScore;
        private Float popularityScore;
        private String academicCategory;
        private Integer rank;
        private LocalDateTime createdAt;
        
        // Helper method to get score value
        public Double getScore() {
            return score != null ? score : (finalScore != null ? finalScore.doubleValue() : 0.0);
        }
    }
}
