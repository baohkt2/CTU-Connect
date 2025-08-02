package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import com.ctuconnect.dto.AuthorInfo;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

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

    // Getter for reaction type (backwards compatibility)
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

    public void setReaction(InteractionType newReaction) {
        if (newReaction == null) {
            throw new IllegalArgumentException("Interaction type cannot be null");
        }
        this.type = newReaction;
        if (newReaction == InteractionType.REACTION) {
            this.reactionType = newReaction.getReactionType();
        } else {
            this.reactionType = null; // Clear reaction type for non-REACTION interactions
        }
    }

    // Enum for interaction types
    public enum InteractionType {
        LIKE,
        SHARE,
        BOOKMARK,
        VIEW,
        COMMENT,
        REACTION // Add REACTION type
        ;

        public ReactionType getReactionType() {
            if (this == REACTION) {
                return ReactionType.LIKE; // Default to LIKE for REACTION type
            }
            return null; // No reaction type for other interaction types
        }
    }

    // Enum for reaction types (for REACTION interactions)
    public enum ReactionType {
        LIKE,
        LOVE,
        HAHA,
        WOW,
        SAD,
        ANGRY
    }

    // Helper methods
    public boolean isReaction() {
        return this.type == InteractionType.REACTION;
    }

    public boolean isLike() {
        return this.type == InteractionType.LIKE ||
               (this.type == InteractionType.REACTION && this.reactionType == ReactionType.LIKE);
    }

    public boolean isView() {
        return this.type == InteractionType.VIEW;
    }

    public boolean isShare() {
        return this.type == InteractionType.SHARE;
    }

    public boolean isBookmark() {
        return this.type == InteractionType.BOOKMARK;
    }

    public boolean isComment() {
        return this.type == InteractionType.COMMENT;
    }

    // Equals and hashCode
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        InteractionEntity that = (InteractionEntity) o;
        return Objects.equals(id, that.id) &&
               Objects.equals(postId, that.postId) &&
               Objects.equals(author, that.author) &&
               type == that.type &&
               reactionType == that.reactionType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, postId, author, type, reactionType);
    }
}
