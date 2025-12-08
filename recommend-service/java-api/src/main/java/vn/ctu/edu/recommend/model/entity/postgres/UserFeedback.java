package vn.ctu.edu.recommend.model.entity.postgres;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;
import vn.ctu.edu.recommend.model.enums.FeedbackType;

import java.time.LocalDateTime;
import java.util.UUID;

/**
 * User feedback entity for reinforcement learning
 */
@Entity
@Table(name = "user_feedback", indexes = {
    @Index(name = "idx_user_post", columnList = "user_id,post_id"),
    @Index(name = "idx_feedback_type", columnList = "feedback_type"),
    @Index(name = "idx_timestamp", columnList = "timestamp")
})
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserFeedback {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "user_id", nullable = false)
    private String userId;

    @Column(name = "post_id", nullable = false)
    private String postId;

    @Enumerated(EnumType.STRING)
    @Column(name = "feedback_type", nullable = false, length = 50)
    private FeedbackType feedbackType;

    /**
     * Feedback score/value
     * For LIKE: 1.0
     * For COMMENT: 2.0
     * For SHARE: 3.0
     * For DWELL_TIME: seconds
     * For SKIP: -1.0
     * For HIDE: -2.0
     */
    @Column(name = "feedback_value", nullable = false)
    private Float feedbackValue;

    @Column(name = "session_id", length = 100)
    private String sessionId;

    @Column(name = "context", columnDefinition = "JSONB")
    private String context;

    @CreationTimestamp
    @Column(name = "timestamp", nullable = false, updatable = false)
    private LocalDateTime timestamp;
}
