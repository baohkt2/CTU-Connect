package com.ctuconnect.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.dto.AuthorInfo;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class PostResponse {
    private String id;
    private String title;
    private String content;
    private AuthorInfo author;
    private String authorId;
    private String authorName;
    private String authorAvatar;
    private List<String> images;
    private List<String> videos; // Added for enhanced functionality
    private List<String> tags;
    private String category;
    private String visibility;
    private String privacy; // For consistency with PostEntity
    private String postType; // Added for enhanced functionality
    private PostEntity.LocationInfo location; // Added for enhanced functionality
    private PostEntity.PostStats stats;
    private PostEntity.EngagementMetrics engagement; // Added for enhanced functionality
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Constructor for backward compatibility
    public PostResponse(PostEntity post) {
        this.id = post.getId();
        this.title = post.getTitle();
        this.content = post.getContent();
        this.author = post.getAuthor();

        // Handle AuthorInfo properly
        if (post.getAuthor() != null) {
            this.authorId = post.getAuthor().getId();
            this.authorName = post.getAuthor().getFullName();
            this.authorAvatar = post.getAuthor().getAvatarUrl();
        }

        this.images = post.getImages();
        this.videos = post.getVideos();
        this.tags = post.getTags();
        this.category = post.getCategory();
        this.visibility = post.getVisibility();
        this.privacy = post.getPrivacy();
        this.postType = post.getPostType() != null ? post.getPostType().name() : null;
        this.location = post.getLocation();
        this.stats = post.getStats();
        this.engagement = post.getEngagement();
        this.createdAt = post.getCreatedAt();
        this.updatedAt = post.getUpdatedAt();
    }
}
