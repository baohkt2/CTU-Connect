package vn.ctu.edu.postservice.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import vn.ctu.edu.postservice.dto.AuthorInfo;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Document(collection = "posts")
public class PostEntity {

    @Id
    private String id;

    private String title;

    private String content;

    @Field("author")
    private AuthorInfo author;

    @Builder.Default
    private List<String> images = new ArrayList<>();

    @Builder.Default
    private List<String> tags = new ArrayList<>();

    private String category;

    @Builder.Default
    private String visibility = "PUBLIC"; // PUBLIC, FRIENDS, PRIVATE

    @Builder.Default
    private PostStats stats = new PostStats();

    @Field("created_at")
    @CreatedDate
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Field("updated_at")
    private LocalDateTime updatedAt;

    // Convenience method to get authorId
    public String getAuthorId() {
        return author != null ? author.getId() : null;
    }

    // Nested class for statistics
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    @Builder
    public static class PostStats {
        @Builder.Default
        private long views = 0;
        @Builder.Default
        private long shares = 0;
        @Builder.Default
        private long comments = 0;
        @Builder.Default
        private Map<InteractionEntity.InteractionType.ReactionType, Integer> reactions = new HashMap<>();

        public void incrementReaction(InteractionEntity.InteractionType.ReactionType reaction) {
            reactions.merge(reaction, 1, Integer::sum);
        }

        public void decrementReaction(InteractionEntity.InteractionType.ReactionType reaction) {
            reactions.computeIfPresent(reaction, (k, v) -> Math.max(0, v - 1));
        }

        public void incrementViews() {
            this.views++;
        }

        public void incrementShares() {
            this.shares++;
        }

        public void incrementComments() {
            this.comments++;
        }

        public void decrementComments() {
            this.comments = Math.max(0, this.comments - 1);
        }

        public void decrementShares() {
            this.shares = Math.max(0, this.shares - 1);
        }

        // Get total likes from all reaction types
        public long getLikes() {
            return reactions.values().stream().mapToInt(Integer::intValue).sum();
        }
    }
}
