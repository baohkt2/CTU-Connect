package vn.ctu.edu.recommend.repository.neo4j;

import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.recommend.model.entity.neo4j.UserNode;

import java.util.List;
import java.util.Map;

/**
 * Repository for User graph operations in Neo4j
 */
@Repository
public interface UserGraphRepository extends Neo4jRepository<UserNode, String> {

    /**
     * Calculate graph relation score between user and post authors
     * Returns weighted scores based on relationship types
     */
    @Query("""
        MATCH (u:User {userId: $userId})
        MATCH (p:Post {postId: $postId})
        MATCH (author:User)-[:POSTED]->(p)
        
        OPTIONAL MATCH path1 = (u)-[:FRIEND]-(author)
        OPTIONAL MATCH path2 = (u)-[:MAJOR]->(m:Major)<-[:MAJOR]-(author)
        OPTIONAL MATCH path3 = (u)-[:FACULTY]->(f:Faculty)<-[:FACULTY]-(author)
        OPTIONAL MATCH path4 = (u)-[:BATCH]->(b:Batch)<-[:BATCH]-(author)
        OPTIONAL MATCH path5 = (u)-[:FOLLOWS]->(author)
        
        WITH u, author, p,
             CASE WHEN path1 IS NOT NULL THEN $friendWeight ELSE 0 END AS friendScore,
             CASE WHEN path2 IS NOT NULL THEN $majorWeight ELSE 0 END AS majorScore,
             CASE WHEN path3 IS NOT NULL THEN $facultyWeight ELSE 0 END AS facultyScore,
             CASE WHEN path4 IS NOT NULL THEN $batchWeight ELSE 0 END AS batchScore,
             CASE WHEN path5 IS NOT NULL THEN $followWeight ELSE 0 END AS followScore
        
        RETURN p.postId AS postId,
               author.userId AS authorId,
               (friendScore + majorScore + facultyScore + batchScore + followScore) AS relationScore
        """)
    Map<String, Object> calculateGraphRelationScore(
        @Param("userId") String userId,
        @Param("postId") String postId,
        @Param("friendWeight") double friendWeight,
        @Param("majorWeight") double majorWeight,
        @Param("facultyWeight") double facultyWeight,
        @Param("batchWeight") double batchWeight,
        @Param("followWeight") double followWeight
    );

    /**
     * Batch calculate graph scores for multiple posts
     */
    @Query("""
        MATCH (u:User {userId: $userId})
        MATCH (p:Post) WHERE p.postId IN $postIds
        MATCH (author:User)-[:POSTED]->(p)
        
        OPTIONAL MATCH path1 = (u)-[:FRIEND]-(author)
        OPTIONAL MATCH path2 = (u)-[:MAJOR]->(m:Major)<-[:MAJOR]-(author)
        OPTIONAL MATCH path3 = (u)-[:FACULTY]->(f:Faculty)<-[:FACULTY]-(author)
        OPTIONAL MATCH path4 = (u)-[:BATCH]->(b:Batch)<-[:BATCH]-(author)
        OPTIONAL MATCH path5 = (u)-[:FOLLOWS]->(author)
        
        WITH u, author, p,
             CASE WHEN path1 IS NOT NULL THEN $friendWeight ELSE 0 END AS friendScore,
             CASE WHEN path2 IS NOT NULL THEN $majorWeight ELSE 0 END AS majorScore,
             CASE WHEN path3 IS NOT NULL THEN $facultyWeight ELSE 0 END AS facultyScore,
             CASE WHEN path4 IS NOT NULL THEN $batchWeight ELSE 0 END AS batchScore,
             CASE WHEN path5 IS NOT NULL THEN $followWeight ELSE 0 END AS followScore
        
        RETURN p.postId AS postId,
               author.userId AS authorId,
               (friendScore + majorScore + facultyScore + batchScore + followScore) AS relationScore
        ORDER BY relationScore DESC
        """)
    List<Map<String, Object>> calculateBatchGraphRelationScores(
        @Param("userId") String userId,
        @Param("postIds") List<String> postIds,
        @Param("friendWeight") double friendWeight,
        @Param("majorWeight") double majorWeight,
        @Param("facultyWeight") double facultyWeight,
        @Param("batchWeight") double batchWeight,
        @Param("followWeight") double followWeight
    );

    /**
     * Find users with similar interests
     */
    @Query("""
        MATCH (u:User {userId: $userId})
        MATCH (similar:User)
        WHERE u <> similar
          AND any(interest IN u.interests WHERE interest IN similar.interests)
        RETURN similar.userId AS userId, 
               size([i IN u.interests WHERE i IN similar.interests]) AS commonInterests
        ORDER BY commonInterests DESC
        LIMIT $limit
        """)
    List<Map<String, Object>> findSimilarUsers(@Param("userId") String userId, @Param("limit") int limit);

    /**
     * Get user's social network statistics
     */
    @Query("""
        MATCH (u:User {userId: $userId})
        OPTIONAL MATCH (u)-[:FRIEND]-(friend:User)
        OPTIONAL MATCH (u)-[:FOLLOWS]->(following:User)
        OPTIONAL MATCH (u)<-[:FOLLOWS]-(follower:User)
        OPTIONAL MATCH (u)-[:POSTED]->(post:Post)
        RETURN count(DISTINCT friend) AS friendCount,
               count(DISTINCT following) AS followingCount,
               count(DISTINCT follower) AS followerCount,
               count(DISTINCT post) AS postCount
        """)
    Map<String, Object> getUserNetworkStats(@Param("userId") String userId);

    /**
     * Find posts from user's network
     */
    @Query("""
        MATCH (u:User {userId: $userId})
        MATCH (u)-[:FRIEND|FOLLOWS*1..2]-(connected:User)
        MATCH (connected)-[:POSTED]->(p:Post)
        WHERE p.createdAt >= datetime($since)
        RETURN DISTINCT p.postId AS postId, 
               connected.userId AS authorId,
               p.createdAt AS createdAt
        ORDER BY p.createdAt DESC
        LIMIT $limit
        """)
    List<Map<String, Object>> findPostsFromNetwork(
        @Param("userId") String userId,
        @Param("since") String since,
        @Param("limit") int limit
    );

    /**
     * Find posts liked/shared by user's connections
     */
    @Query("""
        MATCH (u:User {userId: $userId})
        MATCH (u)-[:FRIEND|FOLLOWS]-(connected:User)
        MATCH (p:Post)<-[:LIKED_BY|SHARED_BY]-(connected)
        RETURN p.postId AS postId,
               count(DISTINCT connected) AS engagementCount
        ORDER BY engagementCount DESC
        LIMIT $limit
        """)
    List<Map<String, Object>> findPopularPostsInNetwork(@Param("userId") String userId, @Param("limit") int limit);
}
