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
import java.time.format.DateTimeFormatter;
import java.time.ZoneId;
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
    private final vn.ctu.edu.recommend.client.PostServiceClient postServiceClient;
    private final PostEmbeddingRepository postEmbeddingRepository;
    private final UserFeedbackRepository userFeedbackRepository;
    private final UserGraphRepository userGraphRepository;
    private final RedisCacheService redisCacheService;
    private final UserInteractionProducer interactionProducer;
    private final TrainingDataProducer trainingDataProducer;
    private final vn.ctu.edu.recommend.nlp.EmbeddingService embeddingService;

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
     * @param excludePostIds Client-sent list of already-seen post IDs to exclude from results
     */
    @Transactional(readOnly = true)
    public RecommendationResponse getFeed(String userId, Integer page, Integer size, Set<String> excludePostIds) {
        long startTime = System.currentTimeMillis();
        
        int requestSize = size != null ? size : defaultRecommendationCount;
        boolean isPagination = excludePostIds != null && !excludePostIds.isEmpty();
        
        log.info("Getting feed for user: {}, size: {}, pagination: {}", userId, requestSize, isPagination);

        try {
            // Step 1: Check cache ONLY if this is the first request (no excludePostIds)
            // Skip cache for pagination requests since each has different exclusion list
            if (!isPagination) {
                List<RecommendationResponse.RecommendedPost> cachedFeed = 
                    redisCacheService.getRecommendations(userId, RecommendationResponse.RecommendedPost.class);
                
                if (cachedFeed != null && !cachedFeed.isEmpty()) {
                    log.info("Returning cached feed for user: {} ({} posts)", userId, cachedFeed.size());
                    return buildResponse(userId, cachedFeed, requestSize, startTime, "cached");
                }
            } else {
                log.info("🔄 Pagination request - skipping cache, excluding {} posts", excludePostIds.size());
            }

            // Step 2: Get user academic profile from user-service
            UserAcademicProfile userProfile = userServiceClient.getUserAcademicProfile(userId);
            log.debug("User profile: major={}, faculty={}", userProfile.getMajor(), userProfile.getFaculty());

            // Step 3: Get user interaction history from Neo4j/Postgres
            List<UserInteractionHistory> userHistory = getUserInteractionHistory(userId, 30);
            log.debug("User has {} interactions in last 30 days", userHistory.size());

            // Step 4: Combine exclusions - interaction history + client-sent excludePostIds
            Set<String> allExcludedIds = new HashSet<>();
            allExcludedIds.addAll(userHistory.stream()
                .map(UserInteractionHistory::getPostId)
                .collect(Collectors.toSet()));
            if (excludePostIds != null) {
                allExcludedIds.addAll(excludePostIds);
            }
            
            log.info("🚫 Total exclusions: {} posts ({} from history, {} from client)",
                allExcludedIds.size(), userHistory.size(), 
                excludePostIds != null ? excludePostIds.size() : 0);
            
            // Step 5: Get candidate posts (exclude all seen + client-excluded posts)
            List<CandidatePost> candidatePosts = getCandidatePosts(userId, allExcludedIds, requestSize * 5);
            log.debug("Found {} candidate posts after exclusions", candidatePosts.size());

            if (candidatePosts.isEmpty()) {
                log.warn("No candidate posts available for user: {}", userId);
                return buildEmptyResponse(userId, startTime);
            }

            List<RecommendationResponse.RecommendedPost> finalRecommendations;

            // Step 6: Call Python model service for ML-based ranking
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

            // Step 7: Apply business rules (block list, friend priority, major priority)
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

            // Step 8: Limit to requested size
            finalRecommendations = finalRecommendations.stream()
                .limit(requestSize)
                .collect(Collectors.toList());
            
            // 🔍 DEBUG: Log final recommendations
            log.info("🎯 FINAL RECOMMENDATIONS:");
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

            // Step 9: Cache results ONLY for first request (not pagination)
            // Don't cache paginated results since each request has different exclusions
            if (!isPagination) {
                long cacheTtl = calculateCacheTtl(finalRecommendations.size());
                redisCacheService.cacheRecommendations(
                    userId, 
                    finalRecommendations, 
                    Duration.ofSeconds(cacheTtl)
                );
                log.debug("✅ Cached {} recommendations for user: {}, TTL: {}s", 
                    finalRecommendations.size(), userId, cacheTtl);
            } else {
                log.debug("⏭️  Skipping cache for pagination request");
            }

            // Step 10: Return response
            return buildResponse(userId, finalRecommendations, requestSize, startTime, 
                isPagination ? "paginated" : "computed");

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
                .context(context)  // Now accepts Map<String, Object> directly
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
            .map(fb -> {
                // Convert LocalDateTime to Unix timestamp (milliseconds)
                Long timestamp = fb.getTimestamp() != null ? 
                    fb.getTimestamp().atZone(ZoneId.systemDefault()).toInstant().toEpochMilli() : null;
                
                return UserInteractionHistory.builder()
                    .postId(fb.getPostId())
                    .liked(fb.getFeedbackType() == FeedbackType.LIKE ? 1 : 0)
                    .commented(fb.getFeedbackType() == FeedbackType.COMMENT ? 1 : 0)
                    .shared(fb.getFeedbackType() == FeedbackType.SHARE ? 1 : 0)
                    .viewDuration(0.0)
                    .timestamp(timestamp)
                    .build();
            })
            .collect(Collectors.toList());
    }

    private List<CandidatePost> getCandidatePosts(String userId, Set<String> excludePostIds, int limit) {
        // Try to get posts from last 7 days first
        LocalDateTime since = LocalDateTime.now().minusDays(7);
        List<PostEmbedding> posts = postEmbeddingRepository.findTrendingPosts(since);
        
        // Fallback: If no recent posts, try last 30 days
        if (posts.isEmpty()) {
            log.warn("⚠️ No posts in last 7 days, trying 30 days");
            since = LocalDateTime.now().minusDays(30);
            posts = postEmbeddingRepository.findTrendingPosts(since);
        }
        
        // Fallback: If still empty, get all posts
        if (posts.isEmpty()) {
            log.warn("⚠️ No posts in last 30 days, fetching all posts");
            posts = postEmbeddingRepository.findAll();
        }
        
        log.info("📦 Found {} total posts in database", posts.size());
        
        // Debug: Count posts by filter reason
        long excludedByHistory = posts.stream()
            .filter(post -> excludePostIds.contains(post.getPostId()))
            .count();
        long excludedByOwnPost = posts.stream()
            .filter(post -> userId.equals(post.getAuthorId()))
            .count();
        long availablePosts = posts.stream()
            .filter(post -> !excludePostIds.contains(post.getPostId()))
            .filter(post -> !userId.equals(post.getAuthorId()))
            .count();
            
        log.info("🔍 Post filtering breakdown for user {}:", userId);
        log.info("   - Total posts: {}", posts.size());
        log.info("   - Excluded by interaction history: {}", excludedByHistory);
        log.info("   - Excluded (user's own posts): {}", excludedByOwnPost);
        log.info("   - Available for recommendation: {}", availablePosts);

        // First pass: get posts NOT in history (fresh posts)
        List<PostEmbedding> freshPosts = posts.stream()
            .filter(post -> !excludePostIds.contains(post.getPostId()))
            .filter(post -> !userId.equals(post.getAuthorId()))
            .limit(limit)
            .collect(Collectors.toList());
        
        // If no fresh posts, fallback to already-viewed posts (excluding user's own)
        // This ensures users always see content, even if they've viewed everything
        if (freshPosts.isEmpty()) {
            log.warn("⚠️ No fresh posts available, falling back to previously viewed posts");
            freshPosts = posts.stream()
                .filter(post -> !userId.equals(post.getAuthorId())) // Still exclude own posts
                .limit(limit)
                .collect(Collectors.toList());
            log.info("📦 Fallback: {} posts available for re-recommendation", freshPosts.size());
        }

        return freshPosts.stream()
            .map(post -> {
                CandidatePost candidate = CandidatePost.builder()
                    .postId(post.getPostId())
                    .content(post.getContent())
                    .hashtags(Collections.emptyList())
                    .mediaDescription(post.getMediaDescription())
                    .authorId(post.getAuthorId())
                    .authorMajor(post.getAuthorMajor())
                    .authorFaculty(post.getAuthorFaculty())
                    .likeCount(post.getLikeCount())
                    .commentCount(post.getCommentCount())
                    .shareCount(post.getShareCount())
                    .viewCount(post.getViewCount())
                    .build();
                    
                // Convert LocalDateTime to ISO string for Python service & keep original
                if (post.getCreatedAt() != null) {
                    candidate.setCreatedAtFromDateTime(post.getCreatedAt());
                }
                
                return candidate;
            })
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
                    .createdAt(post.getCreatedAtDateTime())
                    .build();
            })
            .filter(Objects::nonNull)
            .collect(Collectors.toList());
    }

    private List<RecommendationResponse.RecommendedPost> fallbackRanking(
            List<CandidatePost> candidatePosts, int limit) {
        
        log.warn("🔄 Using fallback ranking for {} candidate posts (Python model unavailable)", candidatePosts.size());
        
        // Simple popularity-based ranking as fallback
        // Calculate normalized popularity score with diversity
        return candidatePosts.stream()
            .sorted((p1, p2) -> {
                // Weight: Like=2, Comment=3, Share=4, View=1
                int score1 = (p1.getLikeCount() * 2) + (p1.getCommentCount() * 3) + (p1.getShareCount() * 4) + p1.getViewCount();
                int score2 = (p2.getLikeCount() * 2) + (p2.getCommentCount() * 3) + (p2.getShareCount() * 4) + p2.getViewCount();
                
                // If both have zero engagement, prioritize by recency
                if (score1 == 0 && score2 == 0) {
                    return p2.getCreatedAt().compareTo(p1.getCreatedAt());
                }
                
                return Integer.compare(score2, score1);
            })
            .limit(limit * 2)
            .map(post -> {
                // Calculate popularity score (0.0 - 1.0)
                int engagementScore = (post.getLikeCount() * 2) + 
                                     (post.getCommentCount() * 3) + 
                                     (post.getShareCount() * 4) + 
                                     post.getViewCount();
                
                // Normalize to 0.0-1.0 range using log scale for better distribution
                double normalizedScore = Math.min(1.0, Math.log1p(engagementScore) / 7.0); // log(1+1000) ≈ 7
                
                // Calculate recency boost (newer posts get higher base)
                LocalDateTime postCreatedAt = post.getCreatedAtDateTime();
                double recencyBoost = 0.0;
                long hoursSinceCreation = 0;
                if (postCreatedAt != null) {
                    hoursSinceCreation = Duration.between(postCreatedAt, LocalDateTime.now()).toHours();
                    recencyBoost = Math.max(0.0, 0.3 - (hoursSinceCreation / 240.0)); // Decay over 10 days
                }
                
                // Combine: base(0.2) + popularity(0.6) + recency(0.2)
                double finalScore = 0.2 + (normalizedScore * 0.6) + recencyBoost;
                finalScore = Math.min(1.0, Math.max(0.2, finalScore)); // Range: 0.2 - 1.0
                
                log.debug("Fallback score for post {}: engagement={}, hours={}, score={}", 
                    post.getPostId(), engagementScore, hoursSinceCreation, finalScore);
                
                return RecommendationResponse.RecommendedPost.builder()
                    .postId(post.getPostId())
                    .authorId(post.getAuthorId())
                    .content(post.getContent())
                    .score(finalScore)
                    .popularityScore((float)normalizedScore)
                    .academicCategory(null)
                    .createdAt(post.getCreatedAtDateTime())
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
                    .timestamp(System.currentTimeMillis())
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

    /**
     * Get statistics about posts in the recommendation database
     */
    public Map<String, Object> getStats() {
        long totalPosts = postEmbeddingRepository.count();
        
        // Get posts from last 7 days
        LocalDateTime since7Days = LocalDateTime.now().minusDays(7);
        List<PostEmbedding> recentPosts = postEmbeddingRepository.findTrendingPosts(since7Days);
        
        // Get posts from last 30 days
        LocalDateTime since30Days = LocalDateTime.now().minusDays(30);
        List<PostEmbedding> monthPosts = postEmbeddingRepository.findTrendingPosts(since30Days);
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalPostsInDb", totalPosts);
        stats.put("postsLast7Days", recentPosts.size());
        stats.put("postsLast30Days", monthPosts.size());
        stats.put("timestamp", LocalDateTime.now());
        
        log.info("📊 Stats: total={}, last7days={}, last30days={}", 
            totalPosts, recentPosts.size(), monthPosts.size());
        
        return stats;
    }

    /**
     * Sync posts from post-service to recommendation database
     * Useful when Kafka events are missed
     */
    @Transactional
    public int syncPostsFromPostService(int limit) {
        log.info("🔄 Starting manual post sync from post-service, limit: {}", limit);
        
        int syncedCount = 0;
        try {
            // Get posts from post-service via Feign client
            List<vn.ctu.edu.recommend.model.dto.PostDTO> posts = postServiceClient.getFeedPosts(0, limit);
            
            log.info("📥 Fetched {} posts from post-service", posts.size());
            
            for (vn.ctu.edu.recommend.model.dto.PostDTO post : posts) {
                try {
                    // Skip if already exists
                    if (postEmbeddingRepository.existsByPostId(post.getId())) {
                        log.debug("Post {} already exists, skipping", post.getId());
                        continue;
                    }
                    
                    String content = post.getContent() != null ? post.getContent() : "";
                    
                    // Generate embedding
                    float[] embedding = embeddingService.generateEmbedding(content, post.getId());
                    
                    // Create post embedding entity
                    PostEmbedding postEmbedding = PostEmbedding.builder()
                        .postId(post.getId())
                        .authorId(post.getAuthorId())
                        .content(content)
                        .academicScore(0.0f)
                        .academicCategory("GENERAL")
                        .popularityScore(0.0f)
                        .contentSimilarityScore(0.0f)
                        .graphRelationScore(0.0f)
                        .likeCount(post.getLikeCount() != null ? post.getLikeCount() : 0)
                        .commentCount(post.getCommentCount() != null ? post.getCommentCount() : 0)
                        .shareCount(post.getShareCount() != null ? post.getShareCount() : 0)
                        .viewCount(0)
                        .embeddingUpdatedAt(LocalDateTime.now())
                        .build();
                    
                    postEmbedding.setEmbeddingVectorFromArray(embedding);
                    postEmbeddingRepository.save(postEmbedding);
                    syncedCount++;
                    
                    log.debug("✅ Synced post: {}", post.getId());
                    
                } catch (Exception e) {
                    log.warn("⚠️ Failed to sync post {}: {}", post.getId(), e.getMessage());
                }
            }
            
            // Invalidate all recommendation caches after sync
            redisCacheService.invalidateAllRecommendations();
            
            log.info("✅ Manual sync completed: {} posts synced", syncedCount);
            return syncedCount;
            
        } catch (Exception e) {
            log.error("❌ Error syncing posts: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to sync posts: " + e.getMessage(), e);
        }
    }

    /**
     * Get user interaction statistics (debug endpoint)
     */
    public Map<String, Object> getUserInteractionStats(String userId, int days) {
        LocalDateTime since = LocalDateTime.now().minusDays(days);
        List<UserFeedback> feedbacks = userFeedbackRepository.findRecentFeedbackByUser(userId, since);
        
        // Group by post ID to see unique posts
        Set<String> uniquePostIds = feedbacks.stream()
            .map(UserFeedback::getPostId)
            .collect(Collectors.toSet());
        
        // Group by feedback type
        Map<String, Long> feedbackCounts = feedbacks.stream()
            .collect(Collectors.groupingBy(
                fb -> fb.getFeedbackType().name(),
                Collectors.counting()
            ));
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("userId", userId);
        stats.put("days", days);
        stats.put("totalInteractions", feedbacks.size());
        stats.put("uniquePostsViewed", uniquePostIds.size());
        stats.put("feedbackBreakdown", feedbackCounts);
        stats.put("viewedPostIds", uniquePostIds);
        stats.put("timestamp", LocalDateTime.now());
        
        log.info("📜 User {} history: {} interactions, {} unique posts in {} days", 
            userId, feedbacks.size(), uniquePostIds.size(), days);
        
        return stats;
    }

    /**
     * Clear user interaction history (allows re-recommendation of viewed posts)
     */
    @Transactional
    public int clearUserHistory(String userId) {
        log.info("🗑️ Clearing interaction history for user: {}", userId);
        
        int deletedCount = userFeedbackRepository.deleteByUserId(userId);
        
        log.info("✅ Cleared {} interactions for user: {}", deletedCount, userId);
        
        return deletedCount;
    }
}
