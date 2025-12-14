package vn.ctu.edu.recommend.model.entity.postgres;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * Friend recommendation log entity
 * Tracks all friend suggestions for analytics and feedback learning
 */
@Entity
@Table(name = "friend_recommendation_log", schema = "recommend", indexes = {
    @Index(name = "idx_frl_user_id", columnList = "user_id"),
    @Index(name = "idx_frl_recommended_user", columnList = "recommended_user_id"),
    @Index(name = "idx_frl_shown_at", columnList = "shown_at"),
    @Index(name = "idx_frl_suggestion_type", columnList = "suggestion_type")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FriendRecommendationLog {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "user_id", nullable = false)
    private String userId;

    @Column(name = "recommended_user_id", nullable = false)
    private String recommendedUserId;

    // ========== Scoring Details ==========

    @Column(name = "relevance_score", nullable = false)
    private Float relevanceScore;

    @Column(name = "content_similarity")
    private Float contentSimilarity;

    @Column(name = "mutual_friends_score")
    private Float mutualFriendsScore;

    @Column(name = "academic_score")
    private Float academicScore;

    @Column(name = "activity_score")
    private Float activityScore;

    @Column(name = "recency_score")
    private Float recencyScore;

    // ========== Suggestion Metadata ==========

    @Column(name = "suggestion_type", nullable = false, length = 50)
    @Enumerated(EnumType.STRING)
    private SuggestionType suggestionType;

    @Column(name = "suggestion_reason", columnDefinition = "TEXT")
    private String suggestionReason;

    @Column(name = "rank_position")
    private Integer rankPosition;

    // ========== Funnel Tracking Timestamps ==========

    @CreationTimestamp
    @Column(name = "shown_at")
    private LocalDateTime shownAt;

    @Column(name = "clicked_at")
    private LocalDateTime clickedAt;

    @Column(name = "friend_request_sent_at")
    private LocalDateTime friendRequestSentAt;

    @Column(name = "accepted_at")
    private LocalDateTime acceptedAt;

    @Column(name = "rejected_at")
    private LocalDateTime rejectedAt;

    @Column(name = "dismissed_at")
    private LocalDateTime dismissedAt;

    // ========== Request Context ==========

    @Column(name = "request_source", length = 50)
    private String requestSource;

    @Column(name = "session_id")
    private String sessionId;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    /**
     * Suggestion type enum
     */
    public enum SuggestionType {
        MUTUAL_FRIENDS,
        ACADEMIC_CONNECTION,
        CONTENT_SIMILARITY,
        ACTIVITY_BASED,
        FRIENDS_OF_FRIENDS,
        PROFILE_VIEWER,
        SIMILAR_INTERESTS,
        SAME_BATCH,
        HYBRID
    }

    /**
     * Check if suggestion was interacted with
     */
    public boolean wasInteracted() {
        return clickedAt != null || friendRequestSentAt != null;
    }

    /**
     * Check if suggestion led to friendship
     */
    public boolean ledToFriendship() {
        return acceptedAt != null;
    }

    /**
     * Calculate conversion rate for this suggestion
     */
    public String getFunnelStage() {
        if (acceptedAt != null) return "ACCEPTED";
        if (rejectedAt != null) return "REJECTED";
        if (friendRequestSentAt != null) return "REQUEST_SENT";
        if (clickedAt != null) return "CLICKED";
        if (dismissedAt != null) return "DISMISSED";
        return "SHOWN";
    }
}
