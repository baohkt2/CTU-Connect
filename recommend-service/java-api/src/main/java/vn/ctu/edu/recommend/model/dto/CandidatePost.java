package vn.ctu.edu.recommend.model.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Collections;
import java.util.List;

/**
 * Candidate post for Python model service
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CandidatePost implements Serializable {
    private static final DateTimeFormatter ISO_FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
    
    private String postId;
    private String content;
    
    @Builder.Default
    private List<String> hashtags = Collections.emptyList();
    
    private String mediaDescription;
    private String authorId;
    private String authorMajor;
    private String authorFaculty;
    private String authorBatch;
    
    // Send as ISO string to Python service
    private String createdAt;
    
    // Store original LocalDateTime for internal use
    @JsonIgnore
    private LocalDateTime createdAtDateTime;
    
    @Builder.Default
    private Integer likeCount = 0;
    
    @Builder.Default
    private Integer commentCount = 0;
    
    @Builder.Default
    private Integer shareCount = 0;
    
    @Builder.Default
    private Integer viewCount = 0;
    
    /**
     * Set createdAt from LocalDateTime
     */
    public void setCreatedAtFromDateTime(LocalDateTime dateTime) {
        if (dateTime != null) {
            this.createdAtDateTime = dateTime;
            this.createdAt = dateTime.format(ISO_FORMATTER);
        }
    }
    
    /**
     * Get createdAt as LocalDateTime for internal use
     */
    @JsonIgnore
    public LocalDateTime getCreatedAtDateTime() {
        if (createdAtDateTime != null) {
            return createdAtDateTime;
        }
        // Try to parse from string if needed
        if (createdAt != null) {
            try {
                return LocalDateTime.parse(createdAt, ISO_FORMATTER);
            } catch (Exception e) {
                return null;
            }
        }
        return null;
    }
}
