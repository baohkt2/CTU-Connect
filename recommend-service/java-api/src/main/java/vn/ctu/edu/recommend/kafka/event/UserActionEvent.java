package vn.ctu.edu.recommend.kafka.event;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * User action event for Kafka messaging
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserActionEvent {
    private String eventId;
    private String actionType; // LIKE, COMMENT, SHARE, VIEW
    private String userId;
    private String postId;
    private Object metadata;
    private LocalDateTime timestamp;
}
