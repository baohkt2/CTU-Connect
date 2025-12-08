package vn.ctu.edu.recommend.model.entity.neo4j;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Property;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.util.HashSet;
import java.util.Set;

/**
 * Neo4j User node for graph relationships
 */
@Node("User")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserNode {

    @Id
    private String userId;

    @Property("name")
    private String name;

    @Property("email")
    private String email;

    @Property("faculty")
    private String faculty;

    @Property("major")
    private String major;

    @Property("batch")
    private String batch;

    @Property("studentClass")
    private String studentClass;

    @Property("interests")
    private String[] interests;

    @Property("activityScore")
    private Double activityScore;

    @Relationship(type = "FRIEND", direction = Relationship.Direction.OUTGOING)
    private Set<UserNode> friends = new HashSet<>();

    @Relationship(type = "FOLLOWS", direction = Relationship.Direction.OUTGOING)
    private Set<UserNode> following = new HashSet<>();

    @Relationship(type = "POSTED", direction = Relationship.Direction.OUTGOING)
    private Set<PostNode> posts = new HashSet<>();
}
