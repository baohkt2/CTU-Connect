package vn.ctu.edu.recommend.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

/**
 * DTO for Post data from post-service
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PostDTO {
    private String id;
    private String authorId;
    private String authorName;
    private String authorAvatar;
    private String content;
    private List<String> mediaUrls;
    private List<String> tags;
    private String category;
    private Integer likeCount;
    private Integer commentCount;
    private Integer shareCount;
    private Integer viewCount;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private String visibility;
    private Boolean isEdited;
    private Boolean isDeleted;
}
