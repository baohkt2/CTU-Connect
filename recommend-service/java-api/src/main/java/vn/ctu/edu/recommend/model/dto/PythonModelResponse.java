package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

/**
 * Response from Python ML model service
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PythonModelResponse implements Serializable {
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RankedPost implements Serializable {
        private String postId;
        private Double score;
        private Double contentSimilarity;
        private Double implicitFeedback;
        private String category;
    }
    
    private List<RankedPost> rankedPosts;
    private String modelVersion;
    private Long processingTimeMs;
}
