package vn.ctu.edu.postservice.dto.response;

import vn.ctu.edu.postservice.dto.AuthorInfo;
import vn.ctu.edu.postservice.entity.PostEntity;
import vn.ctu.edu.postservice.entity.InteractionEntity;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public class PostResponse {
    private String id;
    private String title;
    private String content;
    private String authorId;
    private String authorName;
    private String authorAvatar;
    private List<String> images;
    private List<String> tags;
    private String category;
    private String visibility;
    private PostStatsResponse stats;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Constructors
    public PostResponse() {}

    public PostResponse(PostEntity post) {
        this.id = post.getId();
        this.title = post.getTitle();
        this.content = post.getContent();

        // Handle AuthorInfo properly
        if (post.getAuthor() != null) {
            this.authorId = post.getAuthor().getId();
            this.authorName = post.getAuthor().getName();
            this.authorAvatar = post.getAuthor().getAvatar();
        }

        this.images = post.getImages();
        this.tags = post.getTags();
        this.category = post.getCategory();
        this.visibility = post.getVisibility();
        this.stats = new PostStatsResponse(post.getStats());
        this.createdAt = post.getCreatedAt();
        this.updatedAt = post.getUpdatedAt();
    }

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }

    public String getAuthorId() { return authorId; }
    public void setAuthorId(String authorId) { this.authorId = authorId; }

    public String getAuthorName() { return authorName; }
    public void setAuthorName(String authorName) { this.authorName = authorName; }

    public String getAuthorAvatar() { return authorAvatar; }
    public void setAuthorAvatar(String authorAvatar) { this.authorAvatar = authorAvatar; }

    public List<String> getImages() { return images; }
    public void setImages(List<String> images) { this.images = images; }

    public List<String> getTags() { return tags; }
    public void setTags(List<String> tags) { this.tags = tags; }

    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }

    public String getVisibility() { return visibility; }
    public void setVisibility(String visibility) { this.visibility = visibility; }

    public PostStatsResponse getStats() { return stats; }
    public void setStats(PostStatsResponse stats) { this.stats = stats; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    public static class PostStatsResponse {
        private long views;
        private long likes;
        private long shares;
        private long comments;
        private long bookmarks;
        private Map<InteractionEntity.InteractionType.ReactionType, Integer> reactions;

        public PostStatsResponse() {}

        public PostStatsResponse(PostEntity.PostStats stats) {
            this.views = stats.getViews();
            this.shares = stats.getShares();
            this.comments = stats.getComments();
            this.reactions = stats.getReactions();

            // Calculate total likes from reactions
            this.likes = reactions.values().stream().mapToInt(Integer::intValue).sum();

            // Set bookmarks if available
            this.bookmarks = reactions.getOrDefault(
                InteractionEntity.InteractionType.ReactionType.BOOKMARK, 0
            );
        }

        // Getters and Setters
        public long getViews() { return views; }
        public void setViews(long views) { this.views = views; }

        public long getLikes() { return likes; }
        public void setLikes(long likes) { this.likes = likes; }

        public long getShares() { return shares; }
        public void setShares(long shares) { this.shares = shares; }

        public long getComments() { return comments; }
        public void setComments(long comments) { this.comments = comments; }

        public long getBookmarks() { return bookmarks; }
        public void setBookmarks(long bookmarks) { this.bookmarks = bookmarks; }

        public Map<InteractionEntity.InteractionType.ReactionType, Integer> getReactions() {
            return reactions;
        }
        public void setReactions(Map<InteractionEntity.InteractionType.ReactionType, Integer> reactions) {
            this.reactions = reactions;
        }
    }
}
