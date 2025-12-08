package vn.ctu.edu.recommend.model.entity.neo4j;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Graph relationship result for scoring
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class GraphRelationship {
    private String userId;
    private String postId;
    private String authorId;
    private String relationshipType;
    private Double weight;
    private Integer pathLength;
}
