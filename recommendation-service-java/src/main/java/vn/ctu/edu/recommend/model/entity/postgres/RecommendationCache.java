package vn.ctu.edu.recommend.model.entity.postgres;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * Recommendation cache entity for storing pre-computed recommendations
 */
@Entity
@Table(name = "recommendation_cache", indexes = {
    @Index(name = "idx_user_cache", columnList = "user_id"),
    @Index(name = "idx_updated_cache", columnList = "updated_at")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RecommendationCache {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "user_id", nullable = false, unique = true)
    private String userId;

    /**
     * Array of recommended post IDs in ranked order
     */
    @Column(name = "post_ids", columnDefinition = "TEXT[]")
    private String[] postIds;

    /**
     * Scores corresponding to each post
     */
    @Column(name = "scores", columnDefinition = "REAL[]")
    private Float[] scores;

    @Column(name = "ab_variant", length = 50)
    private String abVariant;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @Column(name = "expires_at", nullable = false)
    private LocalDateTime expiresAt;
}
