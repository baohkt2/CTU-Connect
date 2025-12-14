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
 * User activity score entity
 * Stores calculated activity metrics for friend recommendation ranking
 */
@Entity
@Table(name = "user_activity_score", schema = "recommend", indexes = {
    @Index(name = "idx_uas_user_id", columnList = "user_id"),
    @Index(name = "idx_uas_activity_score", columnList = "activity_score"),
    @Index(name = "idx_uas_last_activity", columnList = "last_activity_at")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserActivityScore {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "user_id", nullable = false, unique = true)
    private String userId;

    // ========== Activity Counts ==========

    @Column(name = "post_count")
    @Builder.Default
    private Integer postCount = 0;

    @Column(name = "comment_count")
    @Builder.Default
    private Integer commentCount = 0;

    @Column(name = "like_count")
    @Builder.Default
    private Integer likeCount = 0;

    @Column(name = "share_count")
    @Builder.Default
    private Integer shareCount = 0;

    @Column(name = "friend_count")
    @Builder.Default
    private Integer friendCount = 0;

    // ========== Recent Activity (Last 30 Days) ==========

    @Column(name = "posts_last_30_days")
    @Builder.Default
    private Integer postsLast30Days = 0;

    @Column(name = "comments_last_30_days")
    @Builder.Default
    private Integer commentsLast30Days = 0;

    @Column(name = "likes_last_30_days")
    @Builder.Default
    private Integer likesLast30Days = 0;

    // ========== Calculated Scores ==========

    /**
     * Overall activity score (0-1)
     * Higher = more active user
     */
    @Column(name = "activity_score")
    @Builder.Default
    private Double activityScore = 0.0;

    /**
     * Engagement rate (interactions / posts seen)
     */
    @Column(name = "engagement_rate")
    @Builder.Default
    private Double engagementRate = 0.0;

    // ========== Timestamps ==========

    @Column(name = "last_activity_at")
    private LocalDateTime lastActivityAt;

    @Column(name = "last_post_at")
    private LocalDateTime lastPostAt;

    @Column(name = "last_login_at")
    private LocalDateTime lastLoginAt;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    /**
     * Calculate and update activity score based on metrics
     * 
     * Scoring formula:
     * - Post count: 30% weight (capped at 100)
     * - Comment count: 10% weight (capped at 500)
     * - Like count: 5% weight (capped at 1000)
     * - Recent activity (30 days): 50% weight
     * - Recency bonus: 30% of final score
     */
    public void calculateActivityScore() {
        // Base score from activity counts
        double baseScore = (
            Math.min(postCount, 100) * 0.3 +
            Math.min(commentCount, 500) * 0.1 +
            Math.min(likeCount, 1000) * 0.05 +
            Math.min(postsLast30Days, 30) * 0.5
        ) / 100.0;

        // Recency bonus (decays over time)
        double recencyBonus = 0.0;
        if (lastActivityAt != null) {
            long daysSinceLastActivity = java.time.Duration.between(lastActivityAt, LocalDateTime.now()).toDays();
            recencyBonus = Math.max(0, 1 - (daysSinceLastActivity / 30.0));
        }

        // Final score (0-1 range)
        this.activityScore = Math.min(1.0, baseScore * 0.7 + recencyBonus * 0.3);
    }

    /**
     * Increment post count and update timestamps
     */
    public void incrementPostCount() {
        this.postCount++;
        this.postsLast30Days++;
        this.lastPostAt = LocalDateTime.now();
        this.lastActivityAt = LocalDateTime.now();
        calculateActivityScore();
    }

    /**
     * Increment comment count and update timestamps
     */
    public void incrementCommentCount() {
        this.commentCount++;
        this.commentsLast30Days++;
        this.lastActivityAt = LocalDateTime.now();
        calculateActivityScore();
    }

    /**
     * Increment like count and update timestamps
     */
    public void incrementLikeCount() {
        this.likeCount++;
        this.likesLast30Days++;
        this.lastActivityAt = LocalDateTime.now();
        calculateActivityScore();
    }

    /**
     * Check if user is considered active
     */
    public boolean isActive() {
        return activityScore >= 0.3;
    }

    /**
     * Check if user has been active recently
     */
    public boolean isRecentlyActive() {
        if (lastActivityAt == null) return false;
        long daysSinceLastActivity = java.time.Duration.between(lastActivityAt, LocalDateTime.now()).toDays();
        return daysSinceLastActivity <= 7;
    }
}
