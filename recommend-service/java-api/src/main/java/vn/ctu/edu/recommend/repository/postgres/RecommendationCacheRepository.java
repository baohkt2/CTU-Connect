package vn.ctu.edu.recommend.repository.postgres;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.recommend.model.entity.postgres.RecommendationCache;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.UUID;

/**
 * Repository for RecommendationCache
 */
@Repository
public interface RecommendationCacheRepository extends JpaRepository<RecommendationCache, UUID> {

    Optional<RecommendationCache> findByUserId(String userId);

    @Query("SELECT rc FROM RecommendationCache rc WHERE rc.userId = :userId AND rc.expiresAt > :now")
    Optional<RecommendationCache> findValidCacheByUserId(@Param("userId") String userId, @Param("now") LocalDateTime now);

    @Modifying
    @Query("DELETE FROM RecommendationCache rc WHERE rc.expiresAt < :now")
    void deleteExpiredCache(@Param("now") LocalDateTime now);

    boolean existsByUserId(String userId);

    void deleteByUserId(String userId);
}
