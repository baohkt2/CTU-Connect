package vn.ctu.edu.recommend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import vn.ctu.edu.recommend.client.PythonModelServiceClient;
import vn.ctu.edu.recommend.client.UserServiceClient;
import vn.ctu.edu.recommend.kafka.producer.TrainingDataProducer;
import vn.ctu.edu.recommend.kafka.producer.UserInteractionProducer;
import vn.ctu.edu.recommend.model.dto.*;
import vn.ctu.edu.recommend.model.entity.postgres.PostEmbedding;
import vn.ctu.edu.recommend.model.entity.postgres.UserFeedback;
import vn.ctu.edu.recommend.model.enums.FeedbackType;
import vn.ctu.edu.recommend.repository.neo4j.UserGraphRepository;
import vn.ctu.edu.recommend.repository.postgres.PostEmbeddingRepository;
import vn.ctu.edu.recommend.repository.postgres.UserFeedbackRepository;
import vn.ctu.edu.recommend.repository.redis.RedisCacheService;

import java.time.LocalDateTime;
import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Enhanced Recommendation Service using Hybrid Architecture
 * - Java: API layer, orchestration, filtering, caching
 * - Python: ML model, embedding, ranking
 * - Kafka: Event streaming for training pipeline
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class HybridRecommendationService {

    private final PythonModelServiceClient pythonModelService;
    private final UserServiceClient userServiceClient;
    private final PostEmbeddingRepository postEmbeddingRepository;
    private final UserFeedbackRepository userFeedbackRepository;
    private final UserGraphRepository userGraphRepository;
    private final RedisCacheService redisCacheService;
    private final UserInteractionProducer interactionProducer;
    private final TrainingDataProducer trainingDataProducer;

    @Value("${recommendation.python-service.enabled:true}")
    private boolean pythonServiceEnabled;

    @Value("${recommendation.cache.min-ttl:30}")
    private long minCacheTtl;

    @Value("${recommendation.cache.max-ttl:120}")
    private long maxCacheTtl;

    @Value("${recommendation.default-recommendation-count:20}")
    private int defaultRecommendationCount;

    /**
     * Main entry point for getting feed recommendations
     * Follows the hybrid architecture flow
     */
    @Transactional(readOnly = true)
    public RecommendationResponse getFeed(String userId, Integer page, Integer size) {
        long startTime = System.currentTimeMillis();
        
        int requestSize = size != null ? size : defaultRecommendationCount;
        log.info("Getting feed for user: {}, size: {}", userId, requestSize);

        try {
            // Step 1: Check cache first (30-120s TTL)
            List<RecommendationResponse.RecommendedPost> cachedFeed = 
                redisCacheService.getRecommendations(userId, RecommendationResponse.RecommendedPost.class);
            
            if (cachedFeed != null && !cachedFeed.isEmpty()) {
                log.info("Returning cached feed for user: {} ({} posts)", userId, cachedFeed.size());
                return buildResponse(userId, cachedFeed, requestSize, startTime, "cached");
            }

            // Step 2: Get user academic profile from user-service
            UserAcademicProfile userProfile = userServiceClient.getUserAcademicProfile(userId);
            log.debug("User profile: major={}, faculty={}", userProfile.getMajor(), userProfile.getFaculty());

            // Step 3: Get user interaction history from Neo4j/Postgres
            List<UserInteractionHistory> userHistory = getUserInteractionHistory(userId, 30);
            log.debug("User has {} interactions in last 30 days", userHistory.size());

            // Step 4: Get candidate posts (filter already seen, apply business rules)
            Set<String> seenPostIds = userHistory.stream()
                .map(UserInteractionHistory::getPostId)
                .collect(Collectors.toSet());
            
            List<CandidatePost> candidatePosts = getCandidatePosts(userId, seenPostIds, requestSize * 5);
            log.debug("Found {} candidate posts", candidatePosts.size());

            if (candidatePosts.isEmpty()) {
                log.warn("No candidate posts available for user: {}", userId);
                return buildEmptyResponse(userId, startTime);
            }

            List<RecommendationResponse.RecommendedPost> finalRecommendations;

            // Step 5: Call Python model service for ML-based ranking
            log.info("🔍 Python service enabled: {}", pythonServiceEnabled);
            
            if (pythonServiceEnabled) {
                log.info("🤖 Calling Python model service...");
                
                PythonModelRequest modelRequest = PythonModelRequest.builder()
                    .userAcademic(userProfile)
                    .userHistory(userHistory)
                    .candidatePosts(candidatePosts)
                    .topK(requestSize * 2)
                    .build();
                
                log.debug("📋 Request: {} candidate posts, {} user history", 
                    candidatePosts.size(), userHistory.size());

                PythonModelResponse modelResponse = null;
                try {
                    modelResponse = pythonModelService.predictRanking(modelRequest);
                } catch (Exception e) {
                    log.error("❌ Error calling Python model: {}", e.getMessage(), e);
                }
                
                if (modelResponse != null && !modelResponse.getRankedPosts().isEmpty()) {
                    log.info("🤖 Python model returned {} ranked posts", modelResponse.getRankedPosts().size());
                    
                    // 🔍 DEBUG: Log Python model rankings
                    log.info("📊 PYTHON MODEL RANKINGS (Top 5):");
                    modelResponse.getRankedPosts().stream()
                        .limit(5)
                        .forEach(ranked -> {
                            log.info("   • PostID: {} | ML Score: {} | Category: {}", 
                                ranked.getPostId(),
                                String.format("%.4f", ranked.getScore()),
                                ranked.getCategory() != null ? ranked.getCategory() : "N/A"
                            );
                        });
                    
                    finalRecommendations = convertPythonResponse(modelResponse, candidatePosts);
                } else {
                    log.warn("⚠️  Python model service unavailable, using fallback ranking");
                    finalRecommendations = fallbackRanking(candidatePosts, requestSize);
                }
            } else {
                log.info("ℹ️  Python service disabled, using fallback ranking");
                finalRecommendations = fallbackRanking(candidatePosts, requestSize);
            }
            
            // 🔍 DEBUG: Log recommendations after Python/Fallback
            log.info("📦 AFTER PYTHON/FALLBACK (before business rules):");
            log.info("   Total: {} posts", finalRecommendations.size());
            if (!finalRecommendations.isEmpty()) {
                finalRecommendations.stream()
                    .limit(3)
                    .forEach(post -> {
                        log.info("   • PostID: {} | Score: {}", 
                            post.getPostId(),
                            String.format("%.4f", post.getScore() != null ? post.getScore() : 0.0)
                        );
                    });
            }

            // Step 6: Apply business rules (block list, friend priority, major priority)
            finalRecommendations = applyBusinessRules(userId, finalRecommendations, userProfile);
            
            // 🔍 DEBUG: Log recommendations after business rules
            log.info("⚖️  AFTER BUSINESS RULES:");
            log.info("   Total: {} posts", finalRecommendations.size());
            if (!finalRecommendations.isEmpty()) {
                finalRecommendations.stream()
                    .limit(3)
                    .forEach(post -> {
                        log.info("   • PostID: {} | Adjusted Score: {}", 
                            post.getPostId(),
                            String.format("%.4f", post.getScore() != null ? post.getScore() : 0.0)
                        );
                    });
            }

            // Step 7: Limit to requested size
            finalRecommendations = finalRecommendations.stream()
                .limit(requestSize)
                .collect(Collectors.toList());
            
            // 🔍 DEBUG: Log final recommendations before caching
            log.info("🎯 FINAL RECOMMENDATIONS (before caching):");
            log.info("   Total: {} posts", finalRecommendations.size());
            
            if (!finalRecommendations.isEmpty()) {
                log.info("   Top 5 Posts:");
                finalRecommendations.stream()
                    .limit(5)
                    .forEach(post -> {
                        log.info("   • PostID: {} | Score: {} | Category: {}", 
                            post.getPostId(),
                            String.format("%.4f", post.getScore() != null ? post.getScore() : 0.0),
                            post.getAcademicCategory() != null ? post.getAcademicCategory() : "N/A"
                        );
                    });
            }

            // Step 8: Cache results (30-120s TTL)
            long cacheTtl = calculateCacheTtl(finalRecommendations.size());
            redisCacheService.cacheRecommendations(
                userId, 
                finalRecommendations, 
                Duration.ofSeconds(cacheTtl)
            );

            // Step 9: Return response
            return buildResponse(userId, finalRecommendations, requestSize, startTime, "computed");

        } catch (Exception e) {
            log.error("Error generating feed for user: {}", userId, e);
            return buildEmptyResponse(userId, startTime);
        }
    }

    /**
     * Record user interaction and send to Kafka for training pipeline
     */
    @Transactional
    public void recordInteraction(String userId, String postId, FeedbackType feedbackType, 
                                   Double viewDuration, Map<String, Object> context) {
        log.info("Recording interaction: user={}, post={}, type={}", userId, postId, feedbackType);

        try {
            // Save to database
            Float feedbackValue = getFeedbackValue(feedbackType);
            
            UserFeedback feedback = UserFeedback.builder()
                .userId(userId)
                .postId(postId)
                .feedbackType(feedbackType)
                .feedbackValue(feedbackValue)
                .context(context != null ? context.toString() : null)
                .build();

            userFeedbackRepository.save(feedback);

            // Send to Kafka for Python training pipeline
            vn.ctu.edu.recommend.kafka.event.UserActionEvent event = 
                vn.ctu.edu.recommend.kafka.event.UserActionEvent.builder()
                    .userId(userId)
                    .postId(postId)
                    .actionType(feedbackType.name())
                    .timestamp(LocalDateTime.now())
                    .metadata(context)
                    .build();

            switch (feedbackType) {
                case VIEW -> interactionProducer.sendPostViewedEvent(event);
                case LIKE -> interactionProducer.sendPostLikedEvent(event);
                case SHARE -> interactionProducer.sendPostSharedEvent(event);
                case COMMENT -> interactionProducer.sendPostCommentedEvent(event);
            }

            // Send training data sample (format: academic_dataset.json)
            sendTrainingDataSample(userId, postId, feedbackType, viewDuration);

            // Invalidate user cache
            redisCacheService.invalidateRecommendations(userId);

        } catch (Exception e) {
            log.error("Error recording interaction: {}", e.getMessage(), e);
        }
    }

    // ========== Private Helper Methods ==========

    private List<UserInteractionHistory> getUserInteractionHistory(String userId, int days) {
        LocalDateTime since = LocalDateTime.now().minusDays(days);
        List<UserFeedback> feedbacks = userFeedbackRepository.findRecentFeedbackByUser(userId, since);

        return feedbacks.stream()
            .map(fb -> UserInteractionHistory.builder()
                .postId(fb.getPostId())
                .liked(fb.getFeedbackType() == FeedbackType.LIKE ? 1 : 0)
                .commented(fb.getFeedbackType() == FeedbackType.COMMENT ? 1 : 0)
                .shared(fb.getFeedbackType() == FeedbackType.SHARE ? 1 : 0)
                .viewDuration(0.0)
                .timestamp(fb.getTimestamp())
                .build())
            .collect(Collectors.toList());
    }

    private List<CandidatePost> getCandidatePosts(String userId, Set<String> excludePostIds, int limit) {
        LocalDateTime since = LocalDateTime.now().minusDays(7);
        List<PostEmbedding> posts = postEmbeddingRepository.findTrendingPosts(since);

        return posts.stream()
            .filter(post -> !excludePostIds.contains(post.getPostId()))
            .limit(limit)
            .map(post -> CandidatePost.builder()
                .postId(post.getPostId())
                .content(post.getContent())
                .hashtags(Collections.emptyList())
                .mediaDescription(post.getMediaDescription())
                .authorId(post.getAuthorId())
                .authorMajor(post.getAuthorMajor())
                .authorFaculty(post.getAuthorFaculty())
                .createdAt(post.getCreatedAt())
                .likeCount(post.getLikeCount())
                .commentCount(post.getCommentCount())
                .shareCount(post.getShareCount())
                .viewCount(post.getViewCount())
                .build())
            .collect(Collectors.toList());
    }

    private List<RecommendationResponse.RecommendedPost> convertPythonResponse(
            PythonModelResponse pythonResponse, List<CandidatePost> candidatePosts) {
        
        Map<String, CandidatePost> postMap = candidatePosts.stream()
            .collect(Collectors.toMap(CandidatePost::getPostId, p -> p));

        return pythonResponse.getRankedPosts().stream()
            .map(ranked -> {
                CandidatePost post = postMap.get(ranked.getPostId());
                if (post == null) return null;

                return RecommendationResponse.RecommendedPost.builder()
                    .postId(post.getPostId())
                    .authorId(post.getAuthorId())
                    .content(post.getContent())
                    .score(ranked.getScore())
                    .contentSimilarity(ranked.getContentSimilarity() != null ? ranked.getContentSimilarity().floatValue() : null)
                    .graphRelationScore(0.0f)
                    .academicScore(0.0f)
                    .popularityScore(0.0f)
                    .academicCategory(ranked.getCategory())
                    .createdAt(post.getCreatedAt())
                    .build();
            })
            .filter(Objects::nonNull)
            .collect(Collectors.toList());
    }

    private List<RecommendationResponse.RecommendedPost> fallbackRanking(
            List<CandidatePost> candidatePosts, int limit) {
        
        log.info("🔄 Using fallback ranking for {} candidate posts", candidatePosts.size());
        
        // Simple popularity-based ranking as fallback
        // Calculate normalized popularity score
        return candidatePosts.stream()
            .sorted((p1, p2) -> {
                int score1 = (p1.getLikeCount() * 2) + p1.getCommentCount() + (p1.getShareCount() * 3) + p1.getViewCount();
                int score2 = (p2.getLikeCount() * 2) + p2.getCommentCount() + (p2.getShareCount() * 3) + p2.getViewCount();
                return Integer.compare(score2, score1);
            })
            .limit(limit * 2)
            .map(post -> {
                // Calculate popularity score (0.0 - 1.0)
                int engagementScore = (post.getLikeCount() * 2) + 
                                     post.getCommentCount() + 
                                     (post.getShareCount() * 3) + 
                                     post.getViewCount();
                
                // Normalize to 0.0-1.0 range (assuming max engagement ~1000)
                double normalizedScore = Math.min(1.0, engagementScore / 1000.0);
                
                // Add base score to prevent 0
                double finalScore = 0.3 + (normalizedScore * 0.7); // Range: 0.3 - 1.0
                
                log.debug("Fallback score for post {}: engagement={}, normalized={}, final={}", 
                    post.getPostId(), engagementScore, normalizedScore, finalScore);
                
                return RecommendationResponse.RecommendedPost.builder()
                    .postId(post.getPostId())
                    .authorId(post.getAuthorId())
                    .content(post.getContent())
                    .score(finalScore)
                    .popularityScore((float)normalizedScore)
                    .academicCategory(null)
                    .createdAt(post.getCreatedAt())
                    .build();
            })
            .collect(Collectors.toList());
    }

    /**
     * Apply business rules to boost/filter recommendations
     * - Boost posts from same major/faculty
     * - Boost posts from friends
     * - Filter blocked users
     * - Filter spam/low quality content
     */
    private List<RecommendationResponse.RecommendedPost> applyBusinessRules(
            String userId, List<RecommendationResponse.RecommendedPost> posts, UserAcademicProfile userProfile) {
        
        // Get blocked users list (implement in user-service client)
        Set<String> blockedUsers = getBlockedUsers(userId);
        
        // Get friends list from Neo4j
        Set<String> friendIds = getFriends(userId);
        
        // Priority boost for same major/faculty and filter blocked
        return posts.stream()
            .filter(post -> !blockedUsers.contains(post.getAuthorId())) // Filter blocked users
            .map(post -> {
                // Get post author info
                Optional<PostEmbedding> postEmbedding = postEmbeddingRepository.findByPostId(post.getPostId());
                if (postEmbedding.isPresent()) {
                    PostEmbedding pe = postEmbedding.get();
                    double currentScore = post.getScore() != null ? post.getScore() : 0.0;
                    
                    // Boost for same major (+0.2)
                    if (userProfile.getMajor() != null && userProfile.getMajor().equals(pe.getAuthorMajor())) {
                        currentScore += 0.2;
                        log.trace("Same major boost for post {}: +0.2", post.getPostId());
                    }
                    
                    // Boost for same faculty (+0.1)
                    if (userProfile.getFaculty() != null && userProfile.getFaculty().equals(pe.getAuthorFaculty())) {
                        currentScore += 0.1;
                        log.trace("Same faculty boost for post {}: +0.1", post.getPostId());
                    }
                    
                    // Boost for friends (+0.3) - highest priority
                    if (friendIds.contains(pe.getAuthorId())) {
                        currentScore += 0.3;
                        log.trace("Friend boost for post {}: +0.3", post.getPostId());
                    }
                    
                    post.setScore(currentScore);
                }
                return post;
            })
            .sorted((p1, p2) -> {
                double score1 = p1.getScore() != null ? p1.getScore() : 0.0;
                double score2 = p2.getScore() != null ? p2.getScore() : 0.0;
                return Double.compare(score2, score1);
            })
            .collect(Collectors.toList());
    }
    
    /**
     * Get user's blocked list
     */
    private Set<String> getBlockedUsers(String userId) {
        try {
            // TODO: Implement via user-service client or Neo4j query
            return Collections.emptySet();
        } catch (Exception e) {
            log.error("Error fetching blocked users for {}: {}", userId, e.getMessage());
            return Collections.emptySet();
        }
    }
    
    /**
     * Get user's friends from Neo4j graph
     */
    private Set<String> getFriends(String userId) {
        try {
            return userGraphRepository.findFriendIds(userId);
        } catch (Exception e) {
            log.error("Error fetching friends for {}: {}", userId, e.getMessage());
            return Collections.emptySet();
        }
    }

    private void sendTrainingDataSample(String userId, String postId, 
                                        FeedbackType feedbackType, Double viewDuration) {
        try {
            UserAcademicProfile userProfile = userServiceClient.getUserAcademicProfile(userId);
            Optional<PostEmbedding> postOpt = postEmbeddingRepository.findByPostId(postId);
            
            if (postOpt.isPresent()) {
                PostEmbedding post = postOpt.get();
                
                CandidatePost candidatePost = CandidatePost.builder()
                    .postId(post.getPostId())
                    .content(post.getContent())
                    .authorId(post.getAuthorId())
                    .authorMajor(post.getAuthorMajor())
                    .authorFaculty(post.getAuthorFaculty())
                    .likeCount(post.getLikeCount())
                    .commentCount(post.getCommentCount())
                    .shareCount(post.getShareCount())
                    .build();

                UserInteractionHistory interaction = UserInteractionHistory.builder()
                    .postId(postId)
                    .liked(feedbackType == FeedbackType.LIKE ? 1 : 0)
                    .commented(feedbackType == FeedbackType.COMMENT ? 1 : 0)
                    .shared(feedbackType == FeedbackType.SHARE ? 1 : 0)
                    .viewDuration(viewDuration != null ? viewDuration : 0.0)
                    .timestamp(LocalDateTime.now())
                    .build();

                trainingDataProducer.sendTrainingDataSample(userProfile, candidatePost, interaction);
            }
        } catch (Exception e) {
            log.error("Error sending training data sample: {}", e.getMessage());
        }
    }

    private Float getFeedbackValue(FeedbackType feedbackType) {
        return switch (feedbackType) {
            case LIKE -> 1.0f;
            case COMMENT -> 2.0f;
            case SHARE -> 3.0f;
            case SAVE -> 2.5f;
            case VIEW -> 0.5f;
            case CLICK -> 0.8f;
            case SKIP -> -0.5f;
            case HIDE -> -2.0f;
            case REPORT -> -3.0f;
            case DWELL_TIME -> 1.0f;
        };
    }

    /**
     * Calculate adaptive cache TTL based on result quality
     * - More results = longer cache (stable feed)
     * - Fewer results = shorter cache (need fresh data)
     * - Bounds: 30-120 seconds
     */
    private long calculateCacheTtl(int resultSize) {
        if (resultSize >= 50) {
            return maxCacheTtl; // 120s for rich feeds
        }
        if (resultSize <= 10) {
            return minCacheTtl; // 30s for sparse feeds
        }
        // Linear interpolation between min and max
        double ratio = (double) resultSize / 50.0;
        long ttl = (long) (minCacheTtl + (ratio * (maxCacheTtl - minCacheTtl)));
        log.debug("Calculated cache TTL: {}s for {} results", ttl, resultSize);
        return ttl;
    }

    private RecommendationResponse buildResponse(
            String userId, List<RecommendationResponse.RecommendedPost> recommendations,
            int size, long startTime, String source) {
        
        long processingTime = System.currentTimeMillis() - startTime;
        
        return RecommendationResponse.builder()
            .userId(userId)
            .recommendations(recommendations.stream().limit(size).collect(Collectors.toList()))
            .totalCount(recommendations.size())
            .page(0)
            .size(size)
            .abVariant(source)
            .timestamp(LocalDateTime.now())
            .processingTimeMs(processingTime)
            .build();
    }

    private RecommendationResponse buildEmptyResponse(String userId, long startTime) {
        return RecommendationResponse.builder()
            .userId(userId)
            .recommendations(Collections.emptyList())
            .totalCount(0)
            .page(0)
            .size(0)
            .abVariant("empty")
            .timestamp(LocalDateTime.now())
            .processingTimeMs(System.currentTimeMillis() - startTime)
            .build();
    }

    /**
     * Invalidate user cache
     */
    public void invalidateUserCache(String userId) {
        log.info("Invalidating cache for user: {}", userId);
        redisCacheService.invalidateRecommendations(userId);
    }
}
