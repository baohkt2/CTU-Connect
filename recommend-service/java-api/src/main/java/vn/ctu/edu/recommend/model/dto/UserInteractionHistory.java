package vn.ctu.edu.recommend.model.dto;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.time.ZoneId;

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
    private Long timestamp;     // Unix timestamp in milliseconds
    
    @JsonIgnore
    private LocalDateTime timestampDateTime;  // For internal use only
    
    /**
     * Set timestamp from LocalDateTime
     */
    public void setTimestampFromDateTime(LocalDateTime dateTime) {
        if (dateTime != null) {
            this.timestampDateTime = dateTime;
            this.timestamp = dateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        }
    }
}
