package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import com.ctuconnect.dto.AuthorInfo;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@Document(collection = "interactions")
public class InteractionEntity {

    @Id
    private String id;

    @Field("post_id")
    private String postId;

    @Field("author")
    private AuthorInfo author;

    private InteractionType type;

    // For REACTION type interactions, store the specific reaction
    private ReactionType reactionType;

    private Map<String, Object> metadata = new HashMap<>();

    @Field("created_at")
    private LocalDateTime createdAt;

    // Constructor
    public InteractionEntity(String postId, AuthorInfo author, InteractionType type) {
        this.postId = postId;
        this.author = author;
        this.type = type;
        this.createdAt = LocalDateTime.now();
        this.metadata = new HashMap<>();
    }

    // Constructor with reaction type
    public InteractionEntity(String postId, AuthorInfo author, InteractionType type, ReactionType reactionType) {
        this.postId = postId;
        this.author = author;
        this.type = type;
        this.reactionType = reactionType;
        this.createdAt = LocalDateTime.now();
        this.metadata = new HashMap<>();
    }

    // Pre-persist hook
    public void prePersist() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
        }
        if (this.metadata == null) {
            this.metadata = new HashMap<>();
        }
    }

    public String getUserId() {
        return author != null ? author.getId() : null;
    }

    // Enum for interaction types
    public enum InteractionType {
        LIKE,
        SHARE,
        BOOKMARK,
        VIEW,
        COMMENT,
        REACTION
    }

    // Enum for reaction types
    public enum ReactionType {
        LIKE,
        LOVE,
        HAHA,
        WOW,
        SAD,
        ANGRY,
        BOOKMARK
    }

    // Helper methods
    public boolean isReaction() {
        return this.type == InteractionType.REACTION;
    }

    public boolean isLike() {
        return this.type == InteractionType.LIKE || 
               (this.type == InteractionType.REACTION && this.reactionType == ReactionType.LIKE);
    }

    public boolean isBookmark() {
        return this.type == InteractionType.BOOKMARK ||
               (this.type == InteractionType.REACTION && this.reactionType == ReactionType.BOOKMARK);
    }

    public boolean isView() {
        return this.type == InteractionType.VIEW;
    }

    public boolean isShare() {
        return this.type == InteractionType.SHARE;
    }
}
