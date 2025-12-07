package vn.ctu.edu.recommend.model.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * Response DTO for recommendation results
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class RecommendationResponse {

    private String userId;
    private List<RecommendedPost> recommendations;
    private Integer totalCount;
    private Integer page;
    private Integer size;
    private String abVariant;
    private LocalDateTime timestamp;
    private Long processingTimeMs;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class RecommendedPost {
        private String postId;
        private String authorId;
        private String content;
        private Float finalScore;
        private Float contentSimilarity;
        private Float graphRelationScore;
        private Float academicScore;
        private Float popularityScore;
        private String academicCategory;
        private Integer rank;
        private RecommendationExplanation explanation;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RecommendationExplanation {
        private String reason;
        private List<String> factors;
        private Map<String, Object> details;
    }
}
