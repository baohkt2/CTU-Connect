package vn.ctu.edu.recommend.repository.neo4j;

import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.recommend.model.entity.neo4j.PostNode;

import java.util.List;
import java.util.Map;

/**
 * Repository for Post graph operations in Neo4j
 */
@Repository
public interface PostGraphRepository extends Neo4jRepository<PostNode, String> {

    /**
     * Create or update post node
     */
    @Query("""
        MERGE (p:Post {postId: $postId})
        SET p.authorId = $authorId,
            p.content = $content,
            p.category = $category,
            p.tags = $tags,
            p.createdAt = datetime($createdAt)
        RETURN p
        """)
    PostNode upsertPost(
        @Param("postId") String postId,
        @Param("authorId") String authorId,
        @Param("content") String content,
        @Param("category") String category,
        @Param("tags") List<String> tags,
        @Param("createdAt") String createdAt
    );

    /**
     * Create user-post interaction relationships
     */
    @Query("""
        MATCH (u:User {userId: $userId})
        MATCH (p:Post {postId: $postId})
        MERGE (u)-[r:$relationType]->(p)
        SET r.timestamp = datetime($timestamp)
        RETURN count(r)
        """)
    Integer createInteraction(
        @Param("userId") String userId,
        @Param("postId") String postId,
        @Param("relationType") String relationType,
        @Param("timestamp") String timestamp
    );

    /**
     * Find posts with similar tags
     */
    @Query("""
        MATCH (p1:Post {postId: $postId})
        MATCH (p2:Post)
        WHERE p1 <> p2 
          AND any(tag IN p1.tags WHERE tag IN p2.tags)
        RETURN p2.postId AS postId,
               size([t IN p1.tags WHERE t IN p2.tags]) AS commonTags
        ORDER BY commonTags DESC
        LIMIT $limit
        """)
    List<Map<String, Object>> findPostsWithSimilarTags(
        @Param("postId") String postId,
        @Param("limit") int limit
    );
}
