package vn.ctu.edu.postservice.entity;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@Document(collection = "interactions")
public class InteractionEntity {

    @Id
    private String id;

    @Field("post_id")
    private String postId;

    @Field("user_id")
    private String userId;

    private InteractionType type;

    private Map<String, Object> metadata = new HashMap<>();

    @Field("created_at")
    private LocalDateTime createdAt;

    // Constructors
    public InteractionEntity() {
        this.createdAt = LocalDateTime.now();
    }

    public InteractionEntity(String postId, String userId, InteractionType type) {
        this();
        this.postId = postId;
        this.userId = userId;
        this.type = type;
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getPostId() {
        return postId;
    }

    public void setPostId(String postId) {
        this.postId = postId;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public InteractionType getType() {
        return type;
    }

    public void setType(InteractionType type) {
        this.type = type;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public void setMetadata(Map<String, Object> metadata) {
        this.metadata = metadata;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public enum InteractionType {
        VIEW, LIKE, SHARE, BOOKMARK
    }
}
