package vn.ctu.edu.postservice.dto.response;

import vn.ctu.edu.postservice.entity.PostEntity;

import java.time.LocalDateTime;
import java.util.List;

public class PostResponse {

    private String id;
    private String title;
    private String content;
    private String authorId;
    private List<String> images;
    private List<String> tags;
    private String category;
    private PostStatsResponse stats;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Constructors
    public PostResponse() {}

    public PostResponse(PostEntity post) {
        this.id = post.getId();
        this.title = post.getTitle();
        this.content = post.getContent();
        this.authorId = post.getAuthorId();
        this.images = post.getImages();
        this.tags = post.getTags();
        this.category = post.getCategory();
        this.stats = new PostStatsResponse(post.getStats());
        this.createdAt = post.getCreatedAt();
        this.updatedAt = post.getUpdatedAt();
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getAuthorId() {
        return authorId;
    }

    public void setAuthorId(String authorId) {
        this.authorId = authorId;
    }

    public List<String> getImages() {
        return images;
    }

    public void setImages(List<String> images) {
        this.images = images;
    }

    public List<String> getTags() {
        return tags;
    }

    public void setTags(List<String> tags) {
        this.tags = tags;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public PostStatsResponse getStats() {
        return stats;
    }

    public void setStats(PostStatsResponse stats) {
        this.stats = stats;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }

    public static class PostStatsResponse {
        private long views;
        private long likes;
        private long shares;
        private long comments;

        public PostStatsResponse() {}

        public PostStatsResponse(PostEntity.PostStats stats) {
            this.views = stats.getViews();
            this.likes = stats.getLikes();
            this.shares = stats.getShares();
            this.comments = stats.getComments();
        }

        public long getViews() {
            return views;
        }

        public void setViews(long views) {
            this.views = views;
        }

        public long getLikes() {
            return likes;
        }

        public void setLikes(long likes) {
            this.likes = likes;
        }

        public long getShares() {
            return shares;
        }

        public void setShares(long shares) {
            this.shares = shares;
        }

        public long getComments() {
            return comments;
        }

        public void setComments(long comments) {
            this.comments = comments;
        }
    }
}
