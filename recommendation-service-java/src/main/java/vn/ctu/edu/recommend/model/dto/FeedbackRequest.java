package vn.ctu.edu.recommend.model.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import vn.ctu.edu.recommend.model.enums.FeedbackType;

import java.util.Map;

/**
 * Request DTO for user feedback
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FeedbackRequest {

    @NotBlank(message = "User ID is required")
    private String userId;

    @NotBlank(message = "Post ID is required")
    private String postId;

    @NotNull(message = "Feedback type is required")
    private FeedbackType feedbackType;

    private Float feedbackValue;

    private String sessionId;

    private Map<String, Object> context;
}
