package vn.ctu.edu.recommend.model.entity.neo4j;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Property;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

/**
 * Neo4j Post node for content relationships
 */
@Node("Post")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PostNode {

    @Id
    private String postId;

    @Property("authorId")
    private String authorId;

    @Property("content")
    private String content;

    @Property("category")
    private String category;

    @Property("tags")
    private String[] tags;

    @Property("createdAt")
    private LocalDateTime createdAt;

    @Relationship(type = "LIKED_BY", direction = Relationship.Direction.INCOMING)
    private Set<UserNode> likedBy = new HashSet<>();

    @Relationship(type = "COMMENTED_BY", direction = Relationship.Direction.INCOMING)
    private Set<UserNode> commentedBy = new HashSet<>();

    @Relationship(type = "SHARED_BY", direction = Relationship.Direction.INCOMING)
    private Set<UserNode> sharedBy = new HashSet<>();
}
