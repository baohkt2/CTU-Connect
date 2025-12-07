package vn.ctu.edu.recommend.model.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

/**
 * Request DTO for getting personalized recommendations
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RecommendationRequest {

    @NotBlank(message = "User ID is required")
    private String userId;

    @Min(value = 1, message = "Page number must be at least 1")
    private Integer page = 0;

    @Min(value = 1, message = "Size must be at least 1")
    @Max(value = 100, message = "Size must not exceed 100")
    private Integer size = 20;

    private Map<String, Object> context;

    private Boolean includeExplanations = false;

    private String[] excludePostIds;

    private String[] filterCategories;
}
