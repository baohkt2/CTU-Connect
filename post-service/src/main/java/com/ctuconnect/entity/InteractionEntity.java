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
    private InteractionType.ReactionType reactionType;

    private Map<String, Object> metadata = new HashMap<>();

    @Field("created_at")
    private LocalDateTime createdAt;

    // Constructors
    public InteractionEntity() {
        this.createdAt = LocalDateTime.now();
    }

    public InteractionEntity(String postId, AuthorInfo author, InteractionType type) {
        this();
        this.postId = postId;
        this.author = author;
        this.type = type;
    }

    public InteractionType.ReactionType getReaction() {
        if (reactionType != null) {
            return reactionType;
        }
        return InteractionType.ReactionType.NONE;
    }

    // Add getReactionType method that PostService is calling
    public InteractionType.ReactionType getReactionType() {
        return this.reactionType;
    }

    public String getUserId() {
        if (author != null) {
            return author.getId();
        }
        return null;
    }

    public void setReaction(InteractionType newReaction) {
        // Default to LIKE if null
        this.type = Objects.requireNonNullElse(newReaction, InteractionType.LIKE);
    }

    @Getter
    public enum InteractionType {
        VIEW,
        LIKE(ReactionType.LIKE),
        SHARE,
        BOOKMARK(ReactionType.BOOKMARK),
        COMMENT,
        REPLY,
        MENTION,
        REPORT,
        REACTION; // Add REACTION constant that PostService is looking for

        private final ReactionType reactionType;

        InteractionType() {
            this.reactionType = null;
        }

        InteractionType(ReactionType reactionType) {
            this.reactionType = reactionType;
        }

        public enum ReactionType {
            LIKE, LOVE, HAPPY, SAD, ANGRY, BOOKMARK, NONE
        }
    }
}
