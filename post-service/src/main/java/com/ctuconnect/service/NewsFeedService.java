package com.ctuconnect.service;

import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.repository.PostRepository;
import com.ctuconnect.client.UserServiceClient;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class NewsFeedService {

    private final PostRepository postRepository;
    private final MongoTemplate mongoTemplate;
    private final UserServiceClient userServiceClient;
    private final RedisTemplate<String, Object> redisTemplate;

    private static final String FEED_CACHE_PREFIX = "user_feed:";
    private static final int FEED_CACHE_TTL = 1800; // 30 minutes
    private static final int MAX_FEED_SIZE = 50;
    private static final double FRIEND_POST_WEIGHT = 1.0;
    private static final double ENGAGEMENT_WEIGHT = 0.8;
    private static final double RECENCY_WEIGHT = 0.6;
    private static final double RELEVANCE_WEIGHT = 0.7;

    /**
     * Generate personalized news feed using Facebook-like algorithm
     */
    public List<PostResponse> generatePersonalizedFeed(String userId, int page, int size) {
        String cacheKey = FEED_CACHE_PREFIX + userId + ":" + page;

        // Try to get from cache first
        List<PostResponse> cachedFeed = getCachedFeed(cacheKey);
        if (cachedFeed != null && !cachedFeed.isEmpty()) {
            return cachedFeed;
        }

        // Get user's social context
        UserSocialContext socialContext = getUserSocialContext(userId);

        // Fetch candidate posts
        List<PostEntity> candidatePosts = fetchCandidatePosts(socialContext, page * size * 3); // Over-fetch for ranking

        // Apply Facebook-like ranking algorithm
        List<PostEntity> rankedPosts = rankPostsForUser(candidatePosts, socialContext);

        // Convert to response DTOs
        List<PostResponse> feedPosts = rankedPosts.stream()
                .limit(size)
                .map(this::convertToResponse)
                .collect(Collectors.toList());

        // Cache the result
        cacheFeed(cacheKey, feedPosts);

        return feedPosts;
    }

    /**
     * Facebook-style post ranking algorithm
     */
    private List<PostEntity> rankPostsForUser(List<PostEntity> posts, UserSocialContext context) {
        return posts.stream()
                .filter(post -> post.isVisibleToUser(context.getUserId(), context.getFriendIds()))
                .map(post -> {
                    double score = calculatePostScore(post, context);
                    return new ScoredPost(post, score);
                })
                .sorted((a, b) -> Double.compare(b.getScore(), a.getScore()))
                .map(ScoredPost::getPost)
                .collect(Collectors.toList());
    }

    /**
     * Calculate post relevance score based on multiple factors
     */
    private double calculatePostScore(PostEntity post, UserSocialContext context) {
        double score = 0.0;

        // 1. Friend relationship score
        double friendScore = calculateFriendScore(post, context);
        score += friendScore * FRIEND_POST_WEIGHT;

        // 2. Engagement score (likes, comments, shares)
        double engagementScore = post.calculateEngagementScore();
        score += engagementScore * ENGAGEMENT_WEIGHT;

        // 3. Recency score (newer posts get higher scores)
        double recencyScore = calculateRecencyScore(post.getCreatedAt());
        score += recencyScore * RECENCY_WEIGHT;

        // 4. Content relevance (academic interests, tags, etc.)
        double relevanceScore = calculateRelevanceScore(post, context);
        score += relevanceScore * RELEVANCE_WEIGHT;

        // 5. Diversity penalty (avoid showing too many posts from same author)
        double diversityPenalty = calculateDiversityPenalty(post, context);
        score *= (1.0 - diversityPenalty);

        return score;
    }

    private double calculateFriendScore(PostEntity post, UserSocialContext context) {
        String authorId = post.getAuthorId();

        if (context.getCloseInteractionIds().contains(authorId)) {
            return 3.0; // Close friends get highest priority
        } else if (context.getFriendIds().contains(authorId)) {
            return 2.0; // Regular friends
        } else if (context.getSameFacultyIds().contains(authorId)) {
            return 1.5; // Same faculty
        } else if (context.getSameMajorIds().contains(authorId)) {
            return 1.2; // Same major
        } else {
            return 1.0; // Public posts
        }
    }

    private double calculateRecencyScore(LocalDateTime createdAt) {
        long hoursOld = java.time.Duration.between(createdAt, LocalDateTime.now()).toHours();
        return Math.exp(-hoursOld / 12.0); // Exponential decay over 12 hours
    }

    private double calculateRelevanceScore(PostEntity post, UserSocialContext context) {
        double relevance = 0.0;

        // Tag matching
        long matchingTags = post.getTags().stream()
                .mapToLong(tag -> context.getInterestTags().contains(tag) ? 1 : 0)
                .sum();
        relevance += matchingTags * 0.5;

        // Category matching
        if (context.getPreferredCategories().contains(post.getCategory())) {
            relevance += 1.0;
        }

        // Academic context matching
        if (post.getAudienceSettings().getAllowedFaculties().contains(context.getFacultyId()) ||
            post.getAudienceSettings().getAllowedMajors().contains(context.getMajorId())) {
            relevance += 0.8;
        }

        return relevance;
    }

    private double calculateDiversityPenalty(PostEntity post, UserSocialContext context) {
        String authorId = post.getAuthorId();
        int recentPostsByAuthor = context.getRecentAuthorCounts().getOrDefault(authorId, 0);

        // Penalize if we've shown many posts from this author recently
        return Math.min(0.5, recentPostsByAuthor * 0.1);
    }

    private List<PostEntity> fetchCandidatePosts(UserSocialContext context, int limit) {
        Query query = new Query();

        // Build criteria for candidate posts
        Criteria criteria = new Criteria();

        // Time window (last 7 days for active feed)
        LocalDateTime weekAgo = LocalDateTime.now().minusDays(7);
        criteria.and("createdAt").gte(weekAgo);

        // Visibility criteria
        List<Criteria> visibilityCriteria = Arrays.asList(
            Criteria.where("privacy").is("PUBLIC"),
            Criteria.where("privacy").is("FRIENDS").and("author.id").in(context.getFriendIds()),
            Criteria.where("audienceSettings.allowedUsers").in(context.getUserId())
        );
        criteria.orOperator(visibilityCriteria.toArray(new Criteria[0]));

        query.addCriteria(criteria);
        query.limit(limit);

        return mongoTemplate.find(query, PostEntity.class);
    }

    private UserSocialContext getUserSocialContext(String userId) {
        // This would typically call the user service to get social graph data
        // For now, returning mock data structure
        return UserSocialContext.builder()
                .userId(userId)
                .friendIds(userServiceClient.getFriendIds(userId))
                .closeInteractionIds(userServiceClient.getCloseInteractionIds(userId))
                .sameFacultyIds(userServiceClient.getSameFacultyUserIds(userId))
                .sameMajorIds(userServiceClient.getSameMajorUserIds(userId))
                .interestTags(userServiceClient.getUserInterestTags(userId))
                .preferredCategories(userServiceClient.getUserPreferredCategories(userId))
                .facultyId(userServiceClient.getUserFacultyId(userId))
                .majorId(userServiceClient.getUserMajorId(userId))
                .recentAuthorCounts(new HashMap<>())
                .build();
    }

    /**
     * Timeline generation for user profile pages
     */
    public List<PostResponse> generateUserTimeline(String userId, String viewerId, int page, int size) {
        Pageable pageable = PageRequest.of(page, size);

        List<PostEntity> userPosts = postRepository.findByAuthorIdOrderByCreatedAtDesc(userId, pageable);

        // Filter based on privacy settings and viewer permissions
        Set<String> viewerFriends = new HashSet<>(userServiceClient.getFriendIds(viewerId));

        return userPosts.stream()
                .filter(post -> post.isVisibleToUser(viewerId, viewerFriends))
                .map(this::convertToResponse)
                .collect(Collectors.toList());
    }

    /**
     * Trending posts algorithm
     */
    public List<PostResponse> getTrendingPosts(int page, int size) {
        LocalDateTime last24Hours = LocalDateTime.now().minusDays(1);

        Query query = new Query();
        query.addCriteria(Criteria.where("createdAt").gte(last24Hours));
        query.addCriteria(Criteria.where("privacy").is("PUBLIC"));

        List<PostEntity> recentPosts = mongoTemplate.find(query, PostEntity.class);

        // Sort by engagement metrics
        return recentPosts.stream()
                .sorted((a, b) -> Double.compare(b.calculateEngagementScore(), a.calculateEngagementScore()))
                .skip(page * size)
                .limit(size)
                .map(this::convertToResponse)
                .collect(Collectors.toList());
    }

    /**
     * Invalidate user's feed cache when relevant events occur
     */
    public void invalidateUserFeedCache(String userId) {
        String pattern = FEED_CACHE_PREFIX + userId + ":*";
        Set<String> keys = redisTemplate.keys(pattern);
        if (!keys.isEmpty()) {
            redisTemplate.delete(keys);
        }
    }

    /**
     * Batch invalidate feed cache for multiple users (e.g., when a popular post is created)
     */
    public void invalidateFeedCacheForUsers(Set<String> userIds) {
        userIds.forEach(this::invalidateUserFeedCache);
    }

    private List<PostResponse> getCachedFeed(String cacheKey) {
        try {
            return (List<PostResponse>) redisTemplate.opsForValue().get(cacheKey);
        } catch (Exception e) {
            log.warn("Failed to get cached feed: {}", e.getMessage());
            return null;
        }
    }

    private void cacheFeed(String cacheKey, List<PostResponse> feed) {
        try {
            redisTemplate.opsForValue().set(cacheKey, feed, FEED_CACHE_TTL, TimeUnit.SECONDS);
        } catch (Exception e) {
            log.warn("Failed to cache feed: {}", e.getMessage());
        }
    }

    private PostResponse convertToResponse(PostEntity post) {
        // Convert PostEntity to PostResponse DTO
        // Implementation would map all fields appropriately
        return PostResponse.builder()
                .id(post.getId())
                .title(post.getTitle())
                .content(post.getContent())
                .author(post.getAuthor())
                .images(post.getImages())
                .videos(post.getVideos())
                .tags(post.getTags())
                .category(post.getCategory())
                .privacy(post.getPrivacy())
                .postType(post.getPostType().name())
                .location(post.getLocation())
                .stats(post.getStats())
                .engagement(post.getEngagement())
                .createdAt(post.getCreatedAt())
                .updatedAt(post.getUpdatedAt())
                .build();
    }

    // Helper classes
    @Data
    @Builder
    private static class UserSocialContext {
        private String userId;
        private Set<String> friendIds;
        private Set<String> closeInteractionIds;
        private Set<String> sameFacultyIds;
        private Set<String> sameMajorIds;
        private Set<String> interestTags;
        private Set<String> preferredCategories;
        private String facultyId;
        private String majorId;
        private Map<String, Integer> recentAuthorCounts;
    }

    @Data
    @AllArgsConstructor
    private static class ScoredPost {
        private PostEntity post;
        private double score;
    }
}
