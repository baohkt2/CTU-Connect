package vn.ctu.edu.postservice.entity;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Document(collection = "posts")
public class PostEntity {

    @Id
    private String id;

    private String title;

    private String content;

    @Field("author_id")
    private String authorId;

    private List<String> images = new ArrayList<>();

    private List<String> tags = new ArrayList<>();

    private String category;

    private PostStats stats = new PostStats();

    @Field("created_at")
    private LocalDateTime createdAt;

    @Field("updated_at")
    private LocalDateTime updatedAt;

    // Constructors
    public PostEntity() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public PostEntity(String title, String content, String authorId) {
        this();
        this.title = title;
        this.content = content;
        this.authorId = authorId;
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

    public PostStats getStats() {
        return stats;
    }

    public void setStats(PostStats stats) {
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

    public void updateTimestamp() {
        this.updatedAt = LocalDateTime.now();
    }

    // Nested class for statistics
    public static class PostStats {
        private long views = 0;
        private long likes = 0;
        private long shares = 0;
        private long comments = 0;

        public long getViews() {
            return views;
        }

        public void setViews(long views) {
            this.views = views;
        }

        public void incrementViews() {
            this.views++;
        }

        public long getLikes() {
            return likes;
        }

        public void setLikes(long likes) {
            this.likes = likes;
        }

        public void incrementLikes() {
            this.likes++;
        }

        public void decrementLikes() {
            this.likes = Math.max(0, this.likes - 1);
        }

        public long getShares() {
            return shares;
        }

        public void setShares(long shares) {
            this.shares = shares;
        }

        public void incrementShares() {
            this.shares++;
        }

        public long getComments() {
            return comments;
        }

        public void setComments(long comments) {
            this.comments = comments;
        }

        public void incrementComments() {
            this.comments++;
        }

        public void decrementComments() {
            this.comments = Math.max(0, this.comments - 1);
        }
    }
}
