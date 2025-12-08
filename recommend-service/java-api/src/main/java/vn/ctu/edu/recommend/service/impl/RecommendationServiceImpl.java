package vn.ctu.edu.recommend.service.impl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import vn.ctu.edu.recommend.model.dto.*;
import vn.ctu.edu.recommend.model.entity.postgres.PostEmbedding;
import vn.ctu.edu.recommend.model.entity.postgres.UserFeedback;
import vn.ctu.edu.recommend.nlp.AcademicClassifier;
import vn.ctu.edu.recommend.nlp.EmbeddingService;
import vn.ctu.edu.recommend.ranking.RankingEngine;
import vn.ctu.edu.recommend.repository.neo4j.UserGraphRepository;
import vn.ctu.edu.recommend.repository.postgres.PostEmbeddingRepository;
import vn.ctu.edu.recommend.repository.postgres.UserFeedbackRepository;
import vn.ctu.edu.recommend.repository.redis.RedisCacheService;
import vn.ctu.edu.recommend.service.RecommendationService;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Implementation of recommendation service with full AI/NLP/Graph ranking
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class RecommendationServiceImpl implements RecommendationService {

    private final PostEmbeddingRepository postEmbeddingRepository;
    private final UserFeedbackRepository userFeedbackRepository;
    private final UserGraphRepository userGraphRepository;
    private final RedisCacheService redisCacheService;
    private final EmbeddingService embeddingService;
    private final AcademicClassifier academicClassifier;
    private final RankingEngine rankingEngine;

    @Value("${recommendation.default-recommendation-count}")
    private int defaultRecommendationCount;

    @Value("${recommendation.graph-weights.friend}")
    private double friendWeight;

    @Value("${recommendation.graph-weights.same-major}")
    private double majorWeight;

    @Value("${recommendation.graph-weights.same-faculty}")
    private double facultyWeight;

    @Value("${recommendation.graph-weights.same-batch}")
    private double batchWeight;

    @Override
    @Transactional(readOnly = true)
    public RecommendationResponse getRecommendations(RecommendationRequest request) {
        long startTime = System.currentTimeMillis();
        
        String userId = request.getUserId();
        int size = request.getSize() != null ? request.getSize() : defaultRecommendationCount;
        
        log.info("Getting recommendations for user: {}, size: {}", userId, size);

        try {
            // 1. Check cache first
            List<RecommendationResponse.RecommendedPost> cachedRecommendations = 
                redisCacheService.getRecommendations(userId, RecommendationResponse.RecommendedPost.class);
            
            if (cachedRecommendations != null && !cachedRecommendations.isEmpty()) {
                log.info("Returning cached recommendations for user: {}", userId);
                return buildResponse(userId, cachedRecommendations, size, startTime, "cached");
            }

            // 2. Get user interests from feedback history
            List<UserFeedback> userFeedbackHistory = userFeedbackRepository
                .findRecentFeedbackByUser(userId, LocalDateTime.now().minusDays(30));
            
            // 3. Get candidate posts (excluding already seen)
            Set<String> seenPostIds = userFeedbackHistory.stream()
                .map(UserFeedback::getPostId)
                .collect(Collectors.toSet());
            
            if (request.getExcludePostIds() != null) {
                seenPostIds.addAll(Arrays.asList(request.getExcludePostIds()));
            }

            List<PostEmbedding> candidatePosts = getCandidatePosts(seenPostIds, size * 5);

            if (candidatePosts.isEmpty()) {
                log.warn("No candidate posts found for user: {}", userId);
                return buildEmptyResponse(userId, startTime);
            }

            // 4. Calculate user interest vector from feedback
            float[] userInterestVector = calculateUserInterestVector(userFeedbackHistory);

            // 5. Calculate content similarity scores
            Map<String, Float> contentSimilarities = calculateContentSimilarities(
                candidatePosts, userInterestVector);

            // 6. Calculate graph relation scores from Neo4j
            Map<String, Float> graphScores = calculateGraphRelationScores(
                userId, candidatePosts);

            // 7. Rank posts using the ranking engine
            List<RecommendationResponse.RecommendedPost> rankedPosts = 
                rankingEngine.rankPosts(candidatePosts, contentSimilarities, graphScores, size * 2);

            // 8. Apply final filters and personalization
            List<RecommendationResponse.RecommendedPost> finalRecommendations = 
                applyFiltersAndPersonalization(rankedPosts, request, size);

            // 9. Cache results
            redisCacheService.cacheRecommendations(
                userId, 
                finalRecommendations, 
                java.time.Duration.ofMinutes(30)
            );

            // 10. Build and return response
            return buildResponse(userId, finalRecommendations, size, startTime, "computed");

        } catch (Exception e) {
            log.error("Error generating recommendations for user: {}", userId, e);
            return buildEmptyResponse(userId, startTime);
        }
    }

    @Override
    @Transactional
    public void recordFeedback(FeedbackRequest request) {
        log.info("Recording feedback: user={}, post={}, type={}", 
            request.getUserId(), request.getPostId(), request.getFeedbackType());

        try {
            // Calculate feedback value based on type
            Float feedbackValue = request.getFeedbackValue();
            if (feedbackValue == null) {
                feedbackValue = getFeedbackValue(request.getFeedbackType());
            }

            // Save feedback
            UserFeedback feedback = UserFeedback.builder()
                .userId(request.getUserId())
                .postId(request.getPostId())
                .feedbackType(request.getFeedbackType())
                .feedbackValue(feedbackValue)
                .sessionId(request.getSessionId())
                .context(request.getContext() != null ? request.getContext().toString() : null)
                .build();

            userFeedbackRepository.save(feedback);

            // Invalidate user's recommendation cache
            redisCacheService.invalidateRecommendations(request.getUserId());

            // Update post engagement metrics if needed
            updatePostEngagementMetrics(request.getPostId(), request.getFeedbackType());

        } catch (Exception e) {
            log.error("Error recording feedback: {}", e.getMessage(), e);
        }
    }

    @Override
    public void rebuildEmbeddings() {
        log.info("Starting embedding rebuild process");
        
        LocalDateTime threshold = LocalDateTime.now().minusHours(24);
        List<PostEmbedding> postsNeedingUpdate = 
            postEmbeddingRepository.findPostsNeedingEmbeddingUpdate(threshold);

        log.info("Found {} posts needing embedding update", postsNeedingUpdate.size());

        for (PostEmbedding post : postsNeedingUpdate) {
            try {
                // Generate embedding
                float[] embedding = embeddingService.generateEmbedding(
                    post.getContent(), post.getPostId());

                // Classify academic content
                ClassificationResponse classification = 
                    academicClassifier.classify(post.getContent());

                // Update post
                post.setEmbeddingVectorFromArray(embedding);
                post.setAcademicScore(classification.getConfidence());
                post.setAcademicCategory(classification.getCategory());
                post.setEmbeddingUpdatedAt(LocalDateTime.now());

                postEmbeddingRepository.save(post);
                
                log.debug("Updated embedding for post: {}", post.getPostId());

            } catch (Exception e) {
                log.error("Failed to update embedding for post: {}", post.getPostId(), e);
            }
        }

        log.info("Completed embedding rebuild");
    }

    @Override
    public void rebuildRecommendationCache() {
        log.info("Starting recommendation cache rebuild");
        redisCacheService.invalidateAllRecommendations();
        log.info("Completed recommendation cache rebuild");
    }

    @Override
    public void invalidateUserCache(String userId) {
        redisCacheService.invalidateRecommendations(userId);
        log.info("Invalidated cache for user: {}", userId);
    }

    // ========== Private Helper Methods ==========

    private List<PostEmbedding> getCandidatePosts(Set<String> excludePostIds, int limit) {
        // Get recent trending posts
        LocalDateTime since = LocalDateTime.now().minusDays(7);
        List<PostEmbedding> trendingPosts = postEmbeddingRepository.findTrendingPosts(since);

        // Filter out excluded posts
        return trendingPosts.stream()
            .filter(post -> !excludePostIds.contains(post.getPostId()))
            .limit(limit)
            .collect(Collectors.toList());
    }

    private float[] calculateUserInterestVector(List<UserFeedback> feedbackHistory) {
        if (feedbackHistory.isEmpty()) {
            return new float[768]; // Return zero vector
        }

        // Get posts user interacted with positively
        List<String> likedPostIds = feedbackHistory.stream()
            .filter(f -> f.getFeedbackValue() > 0)
            .map(UserFeedback::getPostId)
            .collect(Collectors.toList());

        if (likedPostIds.isEmpty()) {
            return new float[768];
        }

        // Calculate average embedding vector
        float[] avgVector = new float[768];
        int count = 0;

        for (String postId : likedPostIds) {
            Optional<PostEmbedding> postOpt = postEmbeddingRepository.findByPostId(postId);
            if (postOpt.isPresent() && postOpt.get().getEmbeddingVector() != null) {
                float[] vector = postOpt.get().getEmbeddingVectorAsArray();
                if (vector != null) {
                    for (int i = 0; i < avgVector.length && i < vector.length; i++) {
                        avgVector[i] += vector[i];
                    }
                    count++;
                }
            }
        }

        // Average the vectors
        if (count > 0) {
            for (int i = 0; i < avgVector.length; i++) {
                avgVector[i] /= count;
            }
        }

        return avgVector;
    }

    private Map<String, Float> calculateContentSimilarities(
            List<PostEmbedding> candidatePosts, float[] userInterestVector) {
        
        Map<String, Float> similarities = new HashMap<>();

        for (PostEmbedding post : candidatePosts) {
            float[] postVector = post.getEmbeddingVectorAsArray();
            if (postVector != null) {
                float similarity = embeddingService.cosineSimilarity(userInterestVector, postVector);
                similarities.put(post.getPostId(), similarity);
            } else {
                similarities.put(post.getPostId(), 0.0f);
            }
        }

        return similarities;
    }

    private Map<String, Float> calculateGraphRelationScores(
            String userId, List<PostEmbedding> candidatePosts) {
        
        Map<String, Float> graphScores = new HashMap<>();

        List<String> postIds = candidatePosts.stream()
            .map(PostEmbedding::getPostId)
            .collect(Collectors.toList());

        try {
            List<Map<String, Object>> results = userGraphRepository.calculateBatchGraphRelationScores(
                userId, postIds, friendWeight, majorWeight, facultyWeight, batchWeight, 1.0);

            for (Map<String, Object> result : results) {
                String postId = (String) result.get("postId");
                Double score = (Double) result.get("relationScore");
                if (postId != null && score != null) {
                    graphScores.put(postId, score.floatValue());
                }
            }
        } catch (Exception e) {
            log.error("Error calculating graph scores: {}", e.getMessage(), e);
        }

        // Fill in missing scores with 0
        for (String postId : postIds) {
            graphScores.putIfAbsent(postId, 0.0f);
        }

        return graphScores;
    }

    private List<RecommendationResponse.RecommendedPost> applyFiltersAndPersonalization(
            List<RecommendationResponse.RecommendedPost> posts, 
            RecommendationRequest request, 
            int limit) {
        
        List<RecommendationResponse.RecommendedPost> filtered = posts;

        // Apply category filter if specified
        if (request.getFilterCategories() != null && request.getFilterCategories().length > 0) {
            Set<String> allowedCategories = Set.of(request.getFilterCategories());
            filtered = posts.stream()
                .filter(p -> p.getAcademicCategory() == null || 
                             allowedCategories.contains(p.getAcademicCategory()))
                .collect(Collectors.toList());
        }

        // Add explanations if requested
        if (request.getIncludeExplanations()) {
            filtered.forEach(this::addExplanation);
        }

        return filtered.stream().limit(limit).collect(Collectors.toList());
    }

    private void addExplanation(RecommendationResponse.RecommendedPost post) {
        List<String> factors = new ArrayList<>();
        
        if (post.getContentSimilarity() > 0.7) {
            factors.add("Similar to your interests");
        }
        if (post.getGraphRelationScore() > 0.5) {
            factors.add("From your network");
        }
        if (post.getAcademicScore() > 0.7) {
            factors.add("Academic content");
        }
        if (post.getPopularityScore() > 0.6) {
            factors.add("Trending");
        }

        String reason = factors.isEmpty() ? "Recommended for you" : String.join(", ", factors);
        
        post.setExplanation(RecommendationResponse.RecommendationExplanation.builder()
            .reason(reason)
            .factors(factors)
            .build());
    }

    private Float getFeedbackValue(vn.ctu.edu.recommend.model.enums.FeedbackType feedbackType) {
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

    private void updatePostEngagementMetrics(String postId, 
            vn.ctu.edu.recommend.model.enums.FeedbackType feedbackType) {
        Optional<PostEmbedding> postOpt = postEmbeddingRepository.findByPostId(postId);
        if (postOpt.isPresent()) {
            PostEmbedding post = postOpt.get();
            
            switch (feedbackType) {
                case LIKE -> post.setLikeCount(post.getLikeCount() + 1);
                case COMMENT -> post.setCommentCount(post.getCommentCount() + 1);
                case SHARE -> post.setShareCount(post.getShareCount() + 1);
                case VIEW -> post.setViewCount(post.getViewCount() + 1);
            }
            
            post.calculatePopularityScore();
            postEmbeddingRepository.save(post);
        }
    }

    private RecommendationResponse buildResponse(
            String userId, 
            List<RecommendationResponse.RecommendedPost> recommendations,
            int size,
            long startTime,
            String source) {
        
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
        long processingTime = System.currentTimeMillis() - startTime;
        
        return RecommendationResponse.builder()
            .userId(userId)
            .recommendations(Collections.emptyList())
            .totalCount(0)
            .page(0)
            .size(0)
            .abVariant("empty")
            .timestamp(LocalDateTime.now())
            .processingTimeMs(processingTime)
            .build();
    }
}
