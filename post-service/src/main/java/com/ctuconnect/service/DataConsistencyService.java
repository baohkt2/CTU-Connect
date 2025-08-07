package com.ctuconnect.service;

import com.ctuconnect.client.UserServiceClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
@Slf4j
public class DataConsistencyService {

    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final RedisTemplate<String, Object> redisTemplate;
    private final UserServiceClient userServiceClient;
    private final PostService postService;

    private static final String SYNC_EVENTS_TOPIC = "data-sync-events";
    private static final String USER_DATA_SYNC_PREFIX = "user_sync:";
    private static final String POST_DATA_SYNC_PREFIX = "post_sync:";

    /**
     * Handle user profile updates across services
     */
    @Transactional
    public void syncUserProfileUpdate(String userId, UserProfileUpdateEvent event) {
        log.info("Syncing user profile update for userId: {}", userId);

        try {
            // 1. Update cached user data in Redis
            updateUserCache(userId, event);

            // 2. Publish event to Kafka for other services
            kafkaTemplate.send(SYNC_EVENTS_TOPIC, event);

            // 3. Invalidate related caches
            invalidateUserRelatedCaches(userId);

            // 4. Update denormalized data in Post service
            CompletableFuture.runAsync(() -> {
                updateUserDataInPosts(userId, event);
            });

        } catch (Exception e) {
            log.error("Failed to sync user profile update for userId: {}", userId, e);
            // Implement compensation logic
            handleSyncFailure(userId, event);
        }
    }

    /**
     * Handle post creation and ensure data consistency
     */
    @Transactional
    public void syncPostCreation(String postId, PostCreationEvent event) {
        log.info("Syncing post creation for postId: {}", postId);

        try {
            // 1. Cache post data
            cachePostData(postId, event);

            // 2. Update user's post count
            updateUserPostCount(event.getAuthorId(), 1);

            // 3. Invalidate friend feeds
            invalidateFriendFeeds(event.getAuthorId());

            // 4. Trigger friend notifications
            notifyFriendsOfNewPost(event);

        } catch (Exception e) {
            log.error("Failed to sync post creation for postId: {}", postId, e);
            handlePostSyncFailure(postId, event);
        }
    }

    /**
     * Handle post interactions and maintain consistency
     */
    public void syncPostInteraction(String postId, PostInteractionEvent event) {
        log.info("Syncing post interaction for postId: {}", postId);

        try {
            // 1. Update post statistics
            updatePostStatistics(postId, event);

            // 2. Update user interaction history
            updateUserInteractionHistory(event.getUserId(), postId, event.getInteractionType());

            // 3. Invalidate relevant caches
            invalidatePostRelatedCaches(postId, event.getAuthorId());

            // 4. Update feed rankings
            updateFeedRankings(postId, event);

        } catch (Exception e) {
            log.error("Failed to sync post interaction for postId: {}", postId, e);
        }
    }

    /**
     * Eventual consistency checker - runs periodically
     */
    public void performConsistencyCheck() {
        log.info("Starting data consistency check");

        // Check user data consistency
        checkUserDataConsistency();

        // Check post statistics consistency
        checkPostStatisticsConsistency();

        // Check relationship consistency
        checkRelationshipConsistency();

        log.info("Data consistency check completed");
    }

    private void updateUserCache(String userId, UserProfileUpdateEvent event) {
        String cacheKey = USER_DATA_SYNC_PREFIX + userId;
        Map<String, Object> userData = new HashMap<>();
        userData.put("fullName", event.getFullName());
        userData.put("avatarUrl", event.getAvatarUrl());
        userData.put("bio", event.getBio());
        userData.put("lastUpdated", LocalDateTime.now());

        redisTemplate.opsForHash().putAll(cacheKey, userData);
        redisTemplate.expire(cacheKey, 24, TimeUnit.HOURS);
    }

    private void invalidateUserRelatedCaches(String userId) {
        // Invalidate friend suggestions
        redisTemplate.delete("friend_suggestions:" + userId);

        // Invalidate user feed
        Set<String> feedKeys = redisTemplate.keys("user_feed:" + userId + ":*");
        if (!feedKeys.isEmpty()) {
            redisTemplate.delete(feedKeys);
        }

        // Invalidate mutual friends cache
        Set<String> mutualKeys = redisTemplate.keys("mutual_friends:" + userId + ":*");
        if (!mutualKeys.isEmpty()) {
            redisTemplate.delete(mutualKeys);
        }
    }

    private void updateUserDataInPosts(String userId, UserProfileUpdateEvent event) {
        // Update author information in all user's posts
        postService.updateAuthorInfoInPosts(userId, event.getFullName(), event.getAvatarUrl());
    }

    private void cachePostData(String postId, PostCreationEvent event) {
        String cacheKey = POST_DATA_SYNC_PREFIX + postId;
        Map<String, Object> postData = new HashMap<>();
        postData.put("authorId", event.getAuthorId());
        postData.put("authorName", event.getAuthorName());
        postData.put("content", event.getContent());
        postData.put("privacy", event.getPrivacy());
        postData.put("createdAt", event.getCreatedAt());

        redisTemplate.opsForHash().putAll(cacheKey, postData);
        redisTemplate.expire(cacheKey, 6, TimeUnit.HOURS);
    }

    private void updateUserPostCount(String userId, int increment) {
        String countKey = "user_post_count:" + userId;
        redisTemplate.opsForValue().increment(countKey, increment);
        redisTemplate.expire(countKey, 24, TimeUnit.HOURS);
    }

    private void invalidateFriendFeeds(String authorId) {
        // Get author's friends
        Set<String> friendIds = userServiceClient.getFriendIds(authorId);

        // Invalidate each friend's feed cache
        friendIds.forEach(friendId -> {
            Set<String> feedKeys = redisTemplate.keys("user_feed:" + friendId + ":*");
            if (!feedKeys.isEmpty()) {
                redisTemplate.delete(feedKeys);
            }
        });
    }

    private void notifyFriendsOfNewPost(PostCreationEvent event) {
        if ("PUBLIC".equals(event.getPrivacy()) || "FRIENDS".equals(event.getPrivacy())) {
            Set<String> friendIds = userServiceClient.getFriendIds(event.getAuthorId());

            // Create notification event
            FriendPostNotificationEvent notificationEvent = FriendPostNotificationEvent.builder()
                    .postId(event.getPostId())
                    .authorId(event.getAuthorId())
                    .authorName(event.getAuthorName())
                    .friendIds(friendIds)
                    .postContent(event.getContent())
                    .build();

            kafkaTemplate.send("friend-post-notifications", notificationEvent);
        }
    }

    private void updatePostStatistics(String postId, PostInteractionEvent event) {
        String statsKey = "post_stats:" + postId;
        String field = event.getInteractionType().toLowerCase() + "_count";

        redisTemplate.opsForHash().increment(statsKey, field, 1);
        redisTemplate.expire(statsKey, 24, TimeUnit.HOURS);
    }

    private void updateUserInteractionHistory(String userId, String postId, String interactionType) {
        String historyKey = "user_interactions:" + userId;
        Map<String, Object> interaction = new HashMap<>();
        interaction.put("postId", postId);
        interaction.put("type", interactionType);
        interaction.put("timestamp", LocalDateTime.now());

        // Store last 100 interactions
        redisTemplate.opsForList().leftPush(historyKey, interaction);
        redisTemplate.opsForList().trim(historyKey, 0, 99);
        redisTemplate.expire(historyKey, 7, TimeUnit.DAYS);
    }

    private void invalidatePostRelatedCaches(String postId, String authorId) {
        // Invalidate post cache
        redisTemplate.delete(POST_DATA_SYNC_PREFIX + postId);

        // Invalidate author's timeline cache
        Set<String> timelineKeys = redisTemplate.keys("user_timeline:" + authorId + ":*");
        if (!timelineKeys.isEmpty()) {
            redisTemplate.delete(timelineKeys);
        }
    }

    private void updateFeedRankings(String postId, PostInteractionEvent event) {
        // Update engagement score for feed ranking
        String engagementKey = "post_engagement:" + postId;
        double score = calculateEngagementScore(event.getInteractionType());
        redisTemplate.opsForValue().increment(engagementKey, score);
        redisTemplate.expire(engagementKey, 24, TimeUnit.HOURS);
    }

    private double calculateEngagementScore(String interactionType) {
        switch (interactionType.toUpperCase()) {
            case "LIKE": return 1.0;
            case "COMMENT": return 2.0;
            case "SHARE": return 3.0;
            default: return 0.5;
        }
    }

    private void checkUserDataConsistency() {
        // Implementation for checking user data consistency across services
        log.info("Checking user data consistency");
    }

    private void checkPostStatisticsConsistency() {
        // Implementation for checking post statistics consistency
        log.info("Checking post statistics consistency");
    }

    private void checkRelationshipConsistency() {
        // Implementation for checking relationship consistency in Neo4j
        log.info("Checking relationship consistency");
    }

    private void handleSyncFailure(String userId, UserProfileUpdateEvent event) {
        // Implement compensation logic for sync failures
        log.warn("Handling sync failure for user: {}", userId);

        // Store failed event for retry
        String failureKey = "sync_failures:user:" + userId;
        redisTemplate.opsForList().leftPush(failureKey, event);
        redisTemplate.expire(failureKey, 1, TimeUnit.DAYS);
    }

    private void handlePostSyncFailure(String postId, PostCreationEvent event) {
        // Implement compensation logic for post sync failures
        log.warn("Handling post sync failure for post: {}", postId);

        String failureKey = "sync_failures:post:" + postId;
        redisTemplate.opsForList().leftPush(failureKey, event);
        redisTemplate.expire(failureKey, 1, TimeUnit.DAYS);
    }

    // Event classes for data synchronization
    @lombok.Data
    @lombok.Builder
    public static class UserProfileUpdateEvent {
        private String userId;
        private String fullName;
        private String avatarUrl;
        private String bio;
        private LocalDateTime updatedAt;
    }

    @lombok.Data
    @lombok.Builder
    public static class PostCreationEvent {
        private String postId;
        private String authorId;
        private String authorName;
        private String content;
        private String privacy;
        private LocalDateTime createdAt;
    }

    @lombok.Data
    @lombok.Builder
    public static class PostInteractionEvent {
        private String postId;
        private String userId;
        private String authorId;
        private String interactionType;
        private LocalDateTime timestamp;
    }

    @lombok.Data
    @lombok.Builder
    public static class FriendPostNotificationEvent {
        private String postId;
        private String authorId;
        private String authorName;
        private Set<String> friendIds;
        private String postContent;
    }
}
