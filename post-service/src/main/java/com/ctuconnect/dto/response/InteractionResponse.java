package com.ctuconnect.dto.response;

import com.ctuconnect.entity.InteractionEntity;

import java.time.LocalDateTime;
import java.util.Map;

public class InteractionResponse {

    private String id;
    private String postId;
    private String userId;
    private InteractionEntity.InteractionType type;
    private Map<String, Object> metadata;
    private LocalDateTime createdAt;

    // Constructors
    public InteractionResponse() {}

    public InteractionResponse(InteractionEntity interaction) {
        this.id = interaction.getId();
        this.postId = interaction.getPostId();
        this.userId = interaction.getUserId();
        this.type = interaction.getType();
        this.metadata = interaction.getMetadata();
        this.createdAt = interaction.getCreatedAt();
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

    public InteractionEntity.InteractionType getType() {
        return type;
    }

    public void setType(InteractionEntity.InteractionType type) {
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
}
