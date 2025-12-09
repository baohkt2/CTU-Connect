package vn.ctu.edu.recommend.model.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDateTime;
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
    private String postId;
    private String content;
    
    @Builder.Default
    private List<String> hashtags = Collections.emptyList();
    
    private String mediaDescription;
    private String authorId;
    private String authorMajor;
    private String authorFaculty;
    private String authorBatch;
    
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime createdAt;
    
    @Builder.Default
    private Integer likeCount = 0;
    
    @Builder.Default
    private Integer commentCount = 0;
    
    @Builder.Default
    private Integer shareCount = 0;
    
    @Builder.Default
    private Integer viewCount = 0;
}
