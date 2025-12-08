package vn.ctu.edu.recommend.kafka.event;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Post event for Kafka messaging
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class PostEvent {
    private PostData data;
    private String eventType; // POST_CREATED, POST_UPDATED, POST_DELETED
    private String postId;
    private String authorId;
    private Long timestamp;

    /**
     * Nested post data structure
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class PostData {
        private String id;
        private String title;
        private String content;
        private Author author;
        private List<String> images;
        private List<String> videos;
        private List<String> documents;
        private List<String> tags;
        private String category;
        private Stats stats;
        private String privacy;
        private AudienceSettings audienceSettings;
        private Engagement engagement;
        private String postType;
        private String location;
        private String crossPostedTo;
        private LocalDateTime scheduledAt;
        private Object editHistory;
        private LocalDateTime createdAt;
        private LocalDateTime updatedAt;
        private boolean pinned;
        private String authorId;
        private String visibility;
        private boolean edited;
        private boolean scheduled;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Author {
        private String id;
        private String name;
        private String avatar;
        private String fullName;
        private String avatarUrl;
        private String role;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Stats {
        private int views;
        private int shares;
        private int comments;
        private int likes;
        private Object reactions;
        private int totalReactions;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class AudienceSettings {
        private String visibility;
        private List<String> allowedUsers;
        private List<String> blockedUsers;
        private List<String> allowedGroups;
        private List<String> allowedFaculties;
        private List<String> allowedMajors;
        private List<String> allowedBatches;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Engagement {
        private double engagementRate;
        private double recentEngagementScore;
        private LocalDateTime lastEngagementAt;
        private Object hourlyEngagement;
    }

    // Convenience methods to access nested data
    public String getContent() {
        return data != null ? data.getContent() : null;
    }

    public String getCategory() {
        return data != null ? data.getCategory() : null;
    }

    public String[] getTags() {
        if (data != null && data.getTags() != null) {
            return data.getTags().toArray(new String[0]);
        }
        return new String[0];
    }
}
