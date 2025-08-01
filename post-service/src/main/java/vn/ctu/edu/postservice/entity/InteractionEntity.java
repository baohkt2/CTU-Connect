package vn.ctu.edu.postservice.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import vn.ctu.edu.postservice.dto.AuthorInfo;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

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
        if (type.getReactionType() != null) {
            return type.getReactionType();
        }
        return InteractionType.ReactionType.NONE;
    }


    @Getter
    public enum InteractionType {
        VIEW,
        LIKE(ReactionType.NONE), // default type
        SHARE,
        BOOKMARK,
        COMMENT,
        REPLY,
        MENTION,
        REPORT;

        private final ReactionType reactionType;

        InteractionType() {
            this.reactionType = null;
        }

        InteractionType(ReactionType reactionType) {
            this.reactionType = reactionType;
        }

        public enum ReactionType {
            LIKE, LOVE, HAPPY, SAD, ANGRY, NONE
        }
    }

}
