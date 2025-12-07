package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDateTime;
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
    private List<String> hashtags;
    private String mediaDescription;
    private String authorId;
    private String authorMajor;
    private String authorFaculty;
    private LocalDateTime createdAt;
    private Integer likeCount;
    private Integer commentCount;
    private Integer shareCount;
    private Integer viewCount;
}
