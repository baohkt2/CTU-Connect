package vn.ctu.edu.recommend.repository.postgres;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.recommend.model.entity.postgres.PostEmbedding;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

/**
 * Repository for PostEmbedding with pgvector support
 */
@Repository
public interface PostEmbeddingRepository extends JpaRepository<PostEmbedding, UUID> {

    Optional<PostEmbedding> findByPostId(String postId);

    List<PostEmbedding> findByAuthorId(String authorId);

    /**
     * Find similar posts using cosine similarity with pgvector
     * Returns posts ordered by similarity score (1 - cosine_distance)
     */
    @Query(value = """
        SELECT pe.*, 
               1 - (pe.embedding_vector <=> CAST(:queryVector AS vector)) AS similarity_score
        FROM post_embeddings pe
        WHERE pe.post_id != :excludePostId
          AND pe.embedding_vector IS NOT NULL
        ORDER BY pe.embedding_vector <=> CAST(:queryVector AS vector)
        LIMIT :limit
        """, nativeQuery = true)
    List<PostEmbedding> findSimilarPosts(
        @Param("queryVector") String queryVector,
        @Param("excludePostId") String excludePostId,
        @Param("limit") int limit
    );

    /**
     * Find posts by academic score threshold
     */
    @Query("SELECT pe FROM PostEmbedding pe WHERE pe.academicScore >= :minScore ORDER BY pe.academicScore DESC")
    List<PostEmbedding> findByAcademicScoreGreaterThanEqual(@Param("minScore") Float minScore);

    /**
     * Find trending posts by popularity
     */
    @Query("SELECT pe FROM PostEmbedding pe WHERE pe.createdAt >= :since ORDER BY pe.popularityScore DESC")
    List<PostEmbedding> findTrendingPosts(@Param("since") LocalDateTime since);

    /**
     * Find posts by faculty/major
     */
    List<PostEmbedding> findByFacultyOrMajor(String faculty, String major);

    /**
     * Find posts needing embedding update
     */
    @Query("SELECT pe FROM PostEmbedding pe WHERE pe.embeddingUpdatedAt IS NULL OR pe.embeddingUpdatedAt < :threshold")
    List<PostEmbedding> findPostsNeedingEmbeddingUpdate(@Param("threshold") LocalDateTime threshold);

    /**
     * Count posts by academic category
     */
    @Query("SELECT pe.academicCategory, COUNT(pe) FROM PostEmbedding pe GROUP BY pe.academicCategory")
    List<Object[]> countByAcademicCategory();

    /**
     * Update engagement metrics
     */
    @Query("UPDATE PostEmbedding pe SET pe.likeCount = :likeCount, pe.commentCount = :commentCount, " +
           "pe.shareCount = :shareCount, pe.viewCount = :viewCount, pe.popularityScore = :popularityScore " +
           "WHERE pe.postId = :postId")
    void updateEngagementMetrics(
        @Param("postId") String postId,
        @Param("likeCount") Integer likeCount,
        @Param("commentCount") Integer commentCount,
        @Param("shareCount") Integer shareCount,
        @Param("viewCount") Integer viewCount,
        @Param("popularityScore") Float popularityScore
    );

    boolean existsByPostId(String postId);

    void deleteByPostId(String postId);
}
