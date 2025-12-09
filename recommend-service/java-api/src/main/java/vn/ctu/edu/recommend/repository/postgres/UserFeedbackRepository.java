package vn.ctu.edu.recommend.repository.postgres;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;
import vn.ctu.edu.recommend.model.entity.postgres.UserFeedback;
import vn.ctu.edu.recommend.model.enums.FeedbackType;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

/**
 * Repository for UserFeedback
 */
@Repository
public interface UserFeedbackRepository extends JpaRepository<UserFeedback, UUID> {

    List<UserFeedback> findByUserId(String userId);

    List<UserFeedback> findByUserIdAndPostId(String userId, String postId);

    List<UserFeedback> findByPostId(String postId);

    List<UserFeedback> findByFeedbackType(FeedbackType feedbackType);
    
    @Transactional
    @Modifying
    @Query("DELETE FROM UserFeedback uf WHERE uf.postId = :postId")
    int deleteByPostId(@Param("postId") String postId);

    @Query("SELECT uf FROM UserFeedback uf WHERE uf.userId = :userId AND uf.timestamp >= :since")
    List<UserFeedback> findRecentFeedbackByUser(@Param("userId") String userId, @Param("since") LocalDateTime since);

    @Query("SELECT uf.postId, COUNT(uf), SUM(uf.feedbackValue) FROM UserFeedback uf " +
           "WHERE uf.userId = :userId GROUP BY uf.postId")
    List<Object[]> findUserFeedbackStats(@Param("userId") String userId);

    @Query("SELECT uf.postId, AVG(uf.feedbackValue) FROM UserFeedback uf " +
           "WHERE uf.postId IN :postIds GROUP BY uf.postId")
    List<Object[]> getAverageFeedbackForPosts(@Param("postIds") List<String> postIds);
}
