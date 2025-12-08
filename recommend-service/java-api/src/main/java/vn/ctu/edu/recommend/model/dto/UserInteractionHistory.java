package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * User interaction history for Python model service
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserInteractionHistory implements Serializable {
    private String postId;
    private Integer liked;      // 0 or 1
    private Integer commented;  // 0 or 1
    private Integer shared;     // 0 or 1
    private Double viewDuration; // in seconds
    private LocalDateTime timestamp;
}
