package vn.ctu.edu.recommend.repository.postgres;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.recommend.model.entity.postgres.FriendRecommendationLog;
import vn.ctu.edu.recommend.model.entity.postgres.FriendRecommendationLog.SuggestionType;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

/**
 * Repository for FriendRecommendationLog
 * Supports analytics and feedback tracking
 */
@Repository
public interface FriendRecommendationLogRepository extends JpaRepository<FriendRecommendationLog, UUID> {

    /**
     * Find recommendations shown to a user
     */
    List<FriendRecommendationLog> findByUserIdOrderByShownAtDesc(String userId);

    /**
     * Find recommendations for a specific user pair
     */
    List<FriendRecommendationLog> findByUserIdAndRecommendedUserId(String userId, String recommendedUserId);

    /**
     * Find recent recommendations shown to user
     */
    @Query("""
        SELECT frl FROM FriendRecommendationLog frl 
        WHERE frl.userId = :userId 
        AND frl.shownAt >= :since 
        ORDER BY frl.shownAt DESC
        """)
    List<FriendRecommendationLog> findRecentRecommendations(
        @Param("userId") String userId,
        @Param("since") LocalDateTime since
    );

    /**
     * Find clicked recommendations
     */
    @Query("""
        SELECT frl FROM FriendRecommendationLog frl 
        WHERE frl.userId = :userId 
        AND frl.clickedAt IS NOT NULL 
        ORDER BY frl.clickedAt DESC
        """)
    List<FriendRecommendationLog> findClickedRecommendations(@Param("userId") String userId);

    /**
     * Find recommendations that led to friend requests
     */
    @Query("""
        SELECT frl FROM FriendRecommendationLog frl 
        WHERE frl.userId = :userId 
        AND frl.friendRequestSentAt IS NOT NULL 
        ORDER BY frl.friendRequestSentAt DESC
        """)
    List<FriendRecommendationLog> findRequestSentRecommendations(@Param("userId") String userId);

    /**
     * Find successful recommendations (accepted friendships)
     */
    @Query("""
        SELECT frl FROM FriendRecommendationLog frl 
        WHERE frl.userId = :userId 
        AND frl.acceptedAt IS NOT NULL 
        ORDER BY frl.acceptedAt DESC
        """)
    List<FriendRecommendationLog> findAcceptedRecommendations(@Param("userId") String userId);

    /**
     * Find recommendations by suggestion type
     */
    List<FriendRecommendationLog> findByUserIdAndSuggestionType(String userId, SuggestionType suggestionType);

    /**
     * Count recommendations by type in time range
     */
    @Query("""
        SELECT frl.suggestionType, COUNT(frl) 
        FROM FriendRecommendationLog frl 
        WHERE frl.shownAt >= :since 
        GROUP BY frl.suggestionType
        """)
    List<Object[]> countBySuggestionTypeInRange(@Param("since") LocalDateTime since);

    /**
     * Calculate click-through rate for a period
     */
    @Query("""
        SELECT 
            COUNT(CASE WHEN frl.clickedAt IS NOT NULL THEN 1 END) * 1.0 / COUNT(*) 
        FROM FriendRecommendationLog frl 
        WHERE frl.shownAt >= :since
        """)
    Double calculateClickThroughRate(@Param("since") LocalDateTime since);

    /**
     * Calculate conversion rate (shown -> accepted)
     */
    @Query("""
        SELECT 
            COUNT(CASE WHEN frl.acceptedAt IS NOT NULL THEN 1 END) * 1.0 / COUNT(*) 
        FROM FriendRecommendationLog frl 
        WHERE frl.shownAt >= :since
        """)
    Double calculateConversionRate(@Param("since") LocalDateTime since);

    /**
     * Find users who have already been recommended to this user recently
     */
    @Query("""
        SELECT frl.recommendedUserId 
        FROM FriendRecommendationLog frl 
        WHERE frl.userId = :userId 
        AND frl.shownAt >= :since
        """)
    List<String> findRecentlyRecommendedUserIds(
        @Param("userId") String userId,
        @Param("since") LocalDateTime since
    );

    /**
     * Find dismissed recommendations
     */
    @Query("""
        SELECT frl.recommendedUserId 
        FROM FriendRecommendationLog frl 
        WHERE frl.userId = :userId 
        AND frl.dismissedAt IS NOT NULL
        """)
    List<String> findDismissedUserIds(@Param("userId") String userId);

    /**
     * Get recommendation stats for a user
     */
    @Query("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN frl.clickedAt IS NOT NULL THEN 1 END) as clicked,
            COUNT(CASE WHEN frl.friendRequestSentAt IS NOT NULL THEN 1 END) as requests,
            COUNT(CASE WHEN frl.acceptedAt IS NOT NULL THEN 1 END) as accepted
        FROM FriendRecommendationLog frl 
        WHERE frl.userId = :userId
        """)
    Object[] getUserRecommendationStats(@Param("userId") String userId);

    /**
     * Delete old logs
     */
    void deleteByShownAtBefore(LocalDateTime threshold);
}
