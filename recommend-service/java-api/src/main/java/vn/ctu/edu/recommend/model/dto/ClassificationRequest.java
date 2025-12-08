package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Request DTO for academic classification
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ClassificationRequest {
    private String text;
    private String model = "academic-classifier";
}
