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

    // Add reactionType field for REACTION interactions
    private ReactionType reactionType;

    private Map<String, Object> metadata = new HashMap<>();

    @Field("created_at")
    private LocalDateTime createdAt;

    // Constructors
    public InteractionEntity(String postId, AuthorInfo author, InteractionType type) {
        this.postId = postId;
        this.author = author;
        this.type = type;
        this.createdAt = LocalDateTime.now();
        this.metadata = new HashMap<>();
    }

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

    // Add convenience methods for checking interaction types
    public boolean isLike() {
        return this.type == InteractionType.LIKE;
    }

    public boolean isReaction() {
        return this.type == InteractionType.REACTION;
    }

    public boolean isBookmark() {
        return this.type == InteractionType.BOOKMARK;
    }

    public boolean isShare() {
        return this.type == InteractionType.SHARE;
    }

    public boolean isView() {
        return this.type == InteractionType.VIEW;
    }

    public void setReaction(InteractionType newReaction) {
        if (newReaction == null) {
            throw new IllegalArgumentException("Interaction type cannot be null");
        }
        this.type = newReaction;
        // Don't automatically set reactionType - it should be set separately when needed
    }

    // Static methods for method references - fix compilation errors
    public static boolean isLike(InteractionEntity entity) {
        return entity != null && entity.isLike();
    }

    public static boolean isBookmark(InteractionEntity entity) {
        return entity != null && entity.isBookmark();
    }

    // Enum for interaction types
    public enum InteractionType {
        LIKE,
        SHARE,
        BOOKMARK,
        VIEW,
        REACTION,
        COMMENT  // Add missing COMMENT enum
    }

    // Separate enum for reaction types - add BOOKMARK
    public enum ReactionType {
        LIKE,
        LOVE,
        HAHA,
        WOW,
        SAD,
        ANGRY,
        BOOKMARK  // Add missing BOOKMARK enum value
    }
}
