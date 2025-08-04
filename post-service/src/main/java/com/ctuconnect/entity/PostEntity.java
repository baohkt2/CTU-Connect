package com.ctuconnect.entity;

import com.ctuconnect.dto.AuthorInfo;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import jakarta.persistence.PrePersist;
import  jakarta.persistence.PreUpdate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Document(collection = "posts")
public class PostEntity {

    @Id
    private String id;

    private String title;

    private String content;

    @Field("author")
    private AuthorInfo author;

    private List<String> images = new ArrayList<>();
    
    private List<String> videos = new ArrayList<>(); // Support for video content

    private List<String> tags = new ArrayList<>();

    private String category;

    private PostStats stats = new PostStats();

    private String privacy;
    
    // Facebook-like audience targeting
    private AudienceSettings audienceSettings = new AudienceSettings();
    
    // Engagement and ranking metrics
    private EngagementMetrics engagement = new EngagementMetrics();
    
    // Post type classification
    private PostType postType = PostType.TEXT;
    
    // Location data
    private LocationInfo location;
    
    // Cross-posting capabilities
    private List<String> crossPostedTo = new ArrayList<>();
    
    // Scheduled posting
    private LocalDateTime scheduledAt;
    
    private boolean isScheduled = false;

    private boolean isPinned = false; // For pinning important posts

    private boolean isEdited = false; // Track if post has been edited
    // Edit history
    private List<EditHistory> editHistory = new ArrayList<>();


    @Field("created_at")
    @CreatedDate
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Field("updated_at")
    private LocalDateTime updatedAt;

    // Pre-persist hook to ensure dates are set correctly
    @PrePersist
    public void prePersist() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
        }
        if (this.updatedAt == null) {
            this.updatedAt = LocalDateTime.now();
        }
    }

    @PreUpdate
    public void preUpdate() {
        this.updatedAt = LocalDateTime.now();
    }

    // Enhanced privacy and visibility methods
    public String getVisibility() {
        return privacy != null ? privacy : "PUBLIC";
    }

    public String getAuthorId() {
        return author != null ? author.getId() : null;
    }

    public void setVisibility(String visibility) {
        if (visibility == null || visibility.isEmpty()) {
            this.privacy = "PUBLIC";
        } else {
            this.privacy = visibility;
        }
    }
    
    // Calculate engagement score for feed ranking
    public double calculateEngagementScore() {
        return engagement.calculateScore();
    }
    
    // Check if post is visible to specific user
    public boolean isVisibleToUser(String userId, Set<String> userFriends) {
        return audienceSettings.isVisibleToUser(userId, userFriends, this.getAuthorId());
    }

    // Nested classes for enhanced functionality
    
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    @Builder
    public static class PostStats {
        @Builder.Default
        private long views = 0;
        @Builder.Default
        private long shares = 0;
        @Builder.Default
        private long comments = 0;
        @Builder.Default
        private long likes = 0;
        @Builder.Default
        private Map<InteractionEntity.ReactionType, Integer> reactions = new HashMap<>();

        public void incrementReaction(InteractionEntity.ReactionType reaction) {
            reactions.merge(reaction, 1, Integer::sum);
            recalculateLikes();
        }

        public void decrementReaction(InteractionEntity.ReactionType reaction) {
            reactions.computeIfPresent(reaction, (k, v) -> Math.max(0, v - 1));
            recalculateLikes();
        }

        public void incrementViews() {
            this.views++;
        }

        public void incrementShares() {
            this.shares++;
        }

        public void incrementComments() {
            this.comments++;
        }

        public void decrementComments() {
            this.comments = Math.max(0, this.comments - 1);
        }

        private void recalculateLikes() {
            this.likes = reactions.values().stream().mapToInt(Integer::intValue).sum();
        }




        public int getTotalReactions() {
            return reactions.values().stream().mapToInt(Integer::intValue).sum();
        }

        public void decrementShares() {
            this.shares = Math.max(0, this.shares - 1);
        }
    }
    
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    @Builder
    public static class AudienceSettings {
        @Builder.Default
        private String visibility = "PUBLIC"; // PUBLIC, FRIENDS, CUSTOM, ONLY_ME
        
        private Set<String> allowedUsers = new HashSet<>();
        private Set<String> blockedUsers = new HashSet<>();
        private Set<String> allowedGroups = new HashSet<>();
        
        // Academic-specific targeting
        private Set<String> allowedFaculties = new HashSet<>();
        private Set<String> allowedMajors = new HashSet<>();
        private Set<String> allowedBatches = new HashSet<>();
        
        public boolean isVisibleToUser(String userId, Set<String> userFriends, String authorId) {
            if (blockedUsers.contains(userId)) return false;
            if (userId.equals(authorId)) return true;
            
            switch (visibility) {
                case "ONLY_ME":
                    return false;
                case "FRIENDS":
                    return userFriends.contains(authorId);
                case "CUSTOM":
                    return allowedUsers.contains(userId);
                case "PUBLIC":
                default:
                    return true;
            }
        }
    }
    
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    @Builder
    public static class EngagementMetrics {
        @Builder.Default
        private double engagementRate = 0.0;
        @Builder.Default
        private double recentEngagementScore = 0.0;
        @Builder.Default
        private LocalDateTime lastEngagementAt = LocalDateTime.now();
        @Builder.Default
        private Map<String, Integer> hourlyEngagement = new HashMap<>();
        
        public double calculateScore() {
            // Facebook-like engagement scoring algorithm
            long hoursOld = java.time.Duration.between(lastEngagementAt, LocalDateTime.now()).toHours();
            double timeDecay = Math.exp(-hoursOld / 24.0); // Decay over 24 hours
            
            return recentEngagementScore * timeDecay;
        }
        
        public void updateEngagement(int likes, int comments, int shares, int views) {
            if (views > 0) {
                this.engagementRate = (double) (likes + comments * 2 + shares * 3) / views;
            }
            this.recentEngagementScore = likes + comments * 2 + shares * 3;
            this.lastEngagementAt = LocalDateTime.now();
        }
    }
    
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    @Builder
    public static class LocationInfo {
        private String placeId;
        private String placeName;
        private double latitude;
        private double longitude;
        private String address;
    }
    
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    @Builder
    public static class EditHistory {
        private String previousContent;
        private LocalDateTime editedAt;
        private String editReason;
    }
    
    public enum PostType {
        TEXT, IMAGE, VIDEO, LINK, POLL, EVENT, SHARED
    }
}
