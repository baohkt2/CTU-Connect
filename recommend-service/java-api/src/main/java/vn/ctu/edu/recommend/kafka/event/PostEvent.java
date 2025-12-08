package vn.ctu.edu.recommend.kafka.event;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * Post event for Kafka messaging
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PostEvent {
    private String eventId;
    private String eventType; // POST_CREATED, POST_UPDATED, POST_DELETED
    private String postId;
    private String authorId;
    private String content;
    private String category;
    private String[] tags;
    private LocalDateTime timestamp;
}
