package vn.ctu.edu.recommend.repository.postgres;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.recommend.model.entity.postgres.UserEmbedding;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository for UserEmbedding with similarity search support
 */
@Repository
public interface UserEmbeddingRepository extends JpaRepository<UserEmbedding, Integer> {

    /**
     * Find embedding by user ID
     */
    Optional<UserEmbedding> findByUserId(String userId);

    /**
     * Check if embedding exists for user
     */
    boolean existsByUserId(String userId);

    /**
     * Find users by faculty
     */
    List<UserEmbedding> findByFaculty(String faculty);

    /**
     * Find users by major
     */
    List<UserEmbedding> findByMajor(String major);

    /**
     * Find users by faculty and major
     */
    List<UserEmbedding> findByFacultyAndMajor(String faculty, String major);

    /**
     * Find users by batch year
     */
    List<UserEmbedding> findByBatchYear(String batchYear);

    /**
     * Find users with embeddings updated recently
     */
    List<UserEmbedding> findByUpdatedAtAfter(LocalDateTime since);

    /**
     * Find users by user IDs
     */
    List<UserEmbedding> findByUserIdIn(List<String> userIds);

    /**
     * Count users with embeddings in a faculty
     */
    @Query("SELECT COUNT(ue) FROM UserEmbedding ue WHERE ue.faculty = :faculty")
    long countByFaculty(@Param("faculty") String faculty);

    /**
     * Find academic connections - same faculty or major
     */
    @Query("""
        SELECT ue FROM UserEmbedding ue 
        WHERE ue.userId != :userId 
        AND (ue.faculty = :faculty OR ue.major = :major)
        ORDER BY 
            CASE WHEN ue.major = :major AND ue.faculty = :faculty THEN 0
                 WHEN ue.major = :major THEN 1
                 WHEN ue.faculty = :faculty THEN 2
                 ELSE 3 END
        """)
    List<UserEmbedding> findAcademicConnections(
        @Param("userId") String userId,
        @Param("faculty") String faculty,
        @Param("major") String major
    );

    /**
     * Delete embedding by user ID
     */
    void deleteByUserId(String userId);

    /**
     * Find stale embeddings (not updated in given days)
     */
    @Query("SELECT ue FROM UserEmbedding ue WHERE ue.updatedAt < :threshold")
    List<UserEmbedding> findStaleEmbeddings(@Param("threshold") LocalDateTime threshold);
}
