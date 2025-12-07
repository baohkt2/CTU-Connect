package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

/**
 * Response DTO from academic classifier
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ClassificationResponse {
    private String category;
    private Float confidence;
    private Map<String, Float> probabilities;
    private Boolean isAcademic;
}
