package vn.ctu.edu.recommend.repository.postgres;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;
import vn.ctu.edu.recommend.model.entity.postgres.UserActivityScore;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

/**
 * Repository for UserActivityScore
 * Manages user activity metrics for friend recommendation
 */
@Repository
public interface UserActivityScoreRepository extends JpaRepository<UserActivityScore, UUID> {

    /**
     * Find activity score by user ID
     */
    Optional<UserActivityScore> findByUserId(String userId);

    /**
     * Check if activity score exists for user
     */
    boolean existsByUserId(String userId);

    /**
     * Find active users (activity score above threshold)
     */
    @Query("SELECT uas FROM UserActivityScore uas WHERE uas.activityScore >= :threshold ORDER BY uas.activityScore DESC")
    List<UserActivityScore> findActiveUsers(@Param("threshold") Double threshold);

    /**
     * Find recently active users
     */
    @Query("SELECT uas FROM UserActivityScore uas WHERE uas.lastActivityAt >= :since ORDER BY uas.lastActivityAt DESC")
    List<UserActivityScore> findRecentlyActiveUsers(@Param("since") LocalDateTime since);

    /**
     * Find top active users
     */
    @Query("SELECT uas FROM UserActivityScore uas ORDER BY uas.activityScore DESC")
    List<UserActivityScore> findTopActiveUsers();

    /**
     * Find users by multiple IDs
     */
    List<UserActivityScore> findByUserIdIn(List<String> userIds);

    /**
     * Get activity scores for multiple users as map
     */
    @Query("SELECT uas FROM UserActivityScore uas WHERE uas.userId IN :userIds")
    List<UserActivityScore> findAllByUserIds(@Param("userIds") List<String> userIds);

    /**
     * Increment post count
     */
    @Modifying
    @Transactional
    @Query("""
        UPDATE UserActivityScore uas 
        SET uas.postCount = uas.postCount + 1,
            uas.postsLast30Days = uas.postsLast30Days + 1,
            uas.lastPostAt = :now,
            uas.lastActivityAt = :now
        WHERE uas.userId = :userId
        """)
    int incrementPostCount(@Param("userId") String userId, @Param("now") LocalDateTime now);

    /**
     * Increment comment count
     */
    @Modifying
    @Transactional
    @Query("""
        UPDATE UserActivityScore uas 
        SET uas.commentCount = uas.commentCount + 1,
            uas.commentsLast30Days = uas.commentsLast30Days + 1,
            uas.lastActivityAt = :now
        WHERE uas.userId = :userId
        """)
    int incrementCommentCount(@Param("userId") String userId, @Param("now") LocalDateTime now);

    /**
     * Increment like count
     */
    @Modifying
    @Transactional
    @Query("""
        UPDATE UserActivityScore uas 
        SET uas.likeCount = uas.likeCount + 1,
            uas.likesLast30Days = uas.likesLast30Days + 1,
            uas.lastActivityAt = :now
        WHERE uas.userId = :userId
        """)
    int incrementLikeCount(@Param("userId") String userId, @Param("now") LocalDateTime now);

    /**
     * Update friend count
     */
    @Modifying
    @Transactional
    @Query("UPDATE UserActivityScore uas SET uas.friendCount = :friendCount WHERE uas.userId = :userId")
    int updateFriendCount(@Param("userId") String userId, @Param("friendCount") Integer friendCount);

    /**
     * Update last login
     */
    @Modifying
    @Transactional
    @Query("UPDATE UserActivityScore uas SET uas.lastLoginAt = :now WHERE uas.userId = :userId")
    int updateLastLogin(@Param("userId") String userId, @Param("now") LocalDateTime now);

    /**
     * Reset 30-day counters (scheduled job)
     */
    @Modifying
    @Transactional
    @Query("""
        UPDATE UserActivityScore uas 
        SET uas.postsLast30Days = 0,
            uas.commentsLast30Days = 0,
            uas.likesLast30Days = 0
        """)
    int resetMonthlyCounters();

    /**
     * Recalculate activity scores for all users (batch job)
     */
    @Query("""
        SELECT uas FROM UserActivityScore uas 
        WHERE uas.updatedAt < :threshold
        """)
    List<UserActivityScore> findScoresNeedingRecalculation(@Param("threshold") LocalDateTime threshold);

    /**
     * Delete by user ID
     */
    void deleteByUserId(String userId);

    /**
     * Get average activity score
     */
    @Query("SELECT AVG(uas.activityScore) FROM UserActivityScore uas")
    Double getAverageActivityScore();

    /**
     * Count active users
     */
    @Query("SELECT COUNT(uas) FROM UserActivityScore uas WHERE uas.activityScore >= :threshold")
    long countActiveUsers(@Param("threshold") Double threshold);
}
