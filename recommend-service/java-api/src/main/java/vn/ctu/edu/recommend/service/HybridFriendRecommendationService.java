package vn.ctu.edu.recommend.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import vn.ctu.edu.recommend.client.PythonModelServiceClient;
import vn.ctu.edu.recommend.client.UserServiceClient;
import vn.ctu.edu.recommend.model.dto.*;
import vn.ctu.edu.recommend.model.entity.postgres.FriendRecommendationLog;
import vn.ctu.edu.recommend.model.entity.postgres.UserActivityScore;
import vn.ctu.edu.recommend.model.entity.postgres.UserEmbedding;
import vn.ctu.edu.recommend.repository.postgres.FriendRecommendationLogRepository;
import vn.ctu.edu.recommend.repository.postgres.UserActivityScoreRepository;
import vn.ctu.edu.recommend.repository.postgres.UserEmbeddingRepository;
import vn.ctu.edu.recommend.repository.redis.RedisCacheService;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Hybrid Friend Recommendation Service
 * 
 * Combines multiple signals for intelligent friend suggestions:
 * - Content Similarity (PhoBERT): 30%
 * - Mutual Friends: 25%
 * - Academic Connection: 20%
 * - Activity Score: 15%
 * - Recency: 10%
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class HybridFriendRecommendationService {

    private final PythonModelServiceClient pythonModelService;
    private final UserServiceClient userServiceClient;
    private final UserEmbeddingRepository userEmbeddingRepository;
    private final UserActivityScoreRepository activityScoreRepository;
    private final FriendRecommendationLogRepository logRepository;
    private final RedisCacheService redisCacheService;

    private static final String FRIEND_SUGGESTION_CACHE_KEY = "friend_suggestions:";
    private static final int DEFAULT_CANDIDATE_LIMIT = 100;

    @Value("${recommendation.friend.enabled:true}")
    private boolean friendRecommendationEnabled;

    @Value("${recommendation.friend.cache-ttl-hours:6}")
    private int cacheTtlHours;

    @Value("${recommendation.friend.default-limit:20}")
    private int defaultLimit;

    @Value("${recommendation.python-service.enabled:true}")
    private boolean pythonServiceEnabled;

    // Scoring weights
    private static final double WEIGHT_CONTENT_SIMILARITY = 0.30;
    private static final double WEIGHT_MUTUAL_FRIENDS = 0.25;
    private static final double WEIGHT_ACADEMIC = 0.20;
    private static final double WEIGHT_ACTIVITY = 0.15;
    private static final double WEIGHT_RECENCY = 0.10;

    /**
     * Main method: Get ML-enhanced friend suggestions
     */
    @Transactional(readOnly = true)
    public FriendRecommendationResponse getFriendSuggestions(String userId, int limit) {
        long startTime = System.currentTimeMillis();
        int requestLimit = limit > 0 ? limit : defaultLimit;
        
        log.info("🤝 Getting friend suggestions for user: {}, limit: {}", userId, requestLimit);

        try {
            // Step 1: Check cache
            List<FriendRecommendationResponse.FriendSuggestion> cached = getCachedSuggestions(userId);
            if (cached != null && !cached.isEmpty()) {
                log.info("📦 Returning cached friend suggestions for user: {} ({} suggestions)", userId, cached.size());
                return buildResponse(userId, cached, requestLimit, startTime, "cache");
            }

            // Step 2: Get current user profile
            UserAcademicProfile currentUserProfile = userServiceClient.getUserAcademicProfile(userId);
            log.debug("Current user: major={}, faculty={}", currentUserProfile.getMajor(), currentUserProfile.getFaculty());

            // Step 3: Get candidate users from User Service
            List<FriendCandidateDTO> candidates = userServiceClient.getFriendCandidates(userId, DEFAULT_CANDIDATE_LIMIT);
            log.debug("Found {} friend candidates", candidates.size());

            if (candidates.isEmpty()) {
                log.warn("No friend candidates found for user: {}", userId);
                return FriendRecommendationResponse.empty(userId);
            }

            // Step 4: Filter out recently shown/dismissed candidates
            Set<String> excludeUserIds = getExcludedUserIds(userId);
            candidates = candidates.stream()
                .filter(c -> !excludeUserIds.contains(c.getUserId()))
                .collect(Collectors.toList());
            
            log.debug("After filtering: {} candidates", candidates.size());

            List<FriendRecommendationResponse.FriendSuggestion> suggestions;

            // Step 5: Use ML or fallback ranking
            if (pythonServiceEnabled && friendRecommendationEnabled) {
                log.info("🤖 Using ML-based friend ranking");
                suggestions = getMLRankedSuggestions(userId, currentUserProfile, candidates, requestLimit);
            } else {
                log.info("📊 Using rule-based friend ranking (ML disabled)");
                suggestions = getRuleBasedSuggestions(userId, currentUserProfile, candidates, requestLimit);
            }

            // Step 6: Cache results
            cacheSuggestions(userId, suggestions);

            // Step 7: Log recommendations for analytics
            logRecommendations(userId, suggestions);

            // Step 8: Return response
            return buildResponse(userId, suggestions, requestLimit, startTime, 
                pythonServiceEnabled ? "ml" : "fallback");

        } catch (Exception e) {
            log.error("Error generating friend suggestions for user: {}", userId, e);
            return FriendRecommendationResponse.empty(userId);
        }
    }

    /**
     * ML-based friend ranking using Python service
     */
    private List<FriendRecommendationResponse.FriendSuggestion> getMLRankedSuggestions(
            String userId,
            UserAcademicProfile currentUser,
            List<FriendCandidateDTO> candidates,
            int limit) {

        try {
            // Build request for Python service
            FriendRankingRequest.UserProfileData currentUserData = FriendRankingRequest.UserProfileData.builder()
                .userId(userId)
                .major(currentUser.getMajor())
                .faculty(currentUser.getFaculty())
                .bio(null)
                .skills(null)
                .courses(null)
                .build();

            List<FriendRankingRequest.UserProfileData> candidateProfiles = candidates.stream()
                .map(c -> FriendRankingRequest.UserProfileData.builder()
                    .userId(c.getUserId())
                    .major(c.getMajorName())
                    .faculty(c.getFacultyName())
                    .bio(c.getBio())
                    .skills(c.getSkills())
                    .courses(c.getCourses())
                    .build())
                .collect(Collectors.toList());

            // Calculate additional scores
            Map<String, FriendRankingRequest.AdditionalScores> additionalScores = calculateAdditionalScores(
                userId, currentUser, candidates
            );

            FriendRankingRequest request = FriendRankingRequest.builder()
                .currentUser(currentUserData)
                .candidates(candidateProfiles)
                .additionalScores(additionalScores)
                .topK(limit * 2) // Request more for filtering
                .build();

            // Call Python service
            FriendRankingResponse response = pythonModelService.rankFriendCandidates(request);

            if (response == null || response.getRankings() == null || response.getRankings().isEmpty()) {
                log.warn("ML ranking returned empty, falling back to rule-based");
                return getRuleBasedSuggestions(userId, currentUser, candidates, limit);
            }

            // Convert to response format
            Map<String, FriendCandidateDTO> candidateMap = candidates.stream()
                .collect(Collectors.toMap(FriendCandidateDTO::getUserId, c -> c));

            return response.getRankings().stream()
                .limit(limit)
                .map(ranked -> {
                    FriendCandidateDTO candidate = candidateMap.get(ranked.getUserId());
                    if (candidate == null) return null;

                    return FriendRecommendationResponse.FriendSuggestion.builder()
                        .userId(candidate.getUserId())
                        .username(candidate.getUsername())
                        .fullName(candidate.getFullName())
                        .avatarUrl(candidate.getAvatarUrl())
                        .bio(candidate.getBio())
                        .facultyName(candidate.getFacultyName())
                        .majorName(candidate.getMajorName())
                        .batchYear(candidate.getBatchYear() != null ? candidate.getBatchYear().toString() : null)
                        .sameFaculty(candidate.isSameFaculty())
                        .sameMajor(candidate.isSameMajor())
                        .sameBatch(candidate.isSameBatch())
                        .mutualFriendsCount(candidate.getMutualFriendsCount())
                        .relevanceScore(ranked.getFinalScore())
                        .contentSimilarity(ranked.getContentSimilarity())
                        .mutualFriendsScore(ranked.getMutualFriendsScore())
                        .academicScore(ranked.getAcademicScore())
                        .activityScore(ranked.getActivityScore())
                        .suggestionType(ranked.getSuggestionType())
                        .suggestionReason(ranked.getSuggestionReason())
                        .build();
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toList());

        } catch (Exception e) {
            log.error("ML friend ranking failed: {}", e.getMessage(), e);
            return getRuleBasedSuggestions(userId, currentUser, candidates, limit);
        }
    }

    /**
     * Rule-based fallback ranking
     */
    private List<FriendRecommendationResponse.FriendSuggestion> getRuleBasedSuggestions(
            String userId,
            UserAcademicProfile currentUser,
            List<FriendCandidateDTO> candidates,
            int limit) {

        log.debug("Applying rule-based ranking for {} candidates", candidates.size());

        return candidates.stream()
            .map(candidate -> {
                // Calculate scores
                double mutualFriendsScore = calculateMutualFriendsScore(candidate.getMutualFriendsCount());
                double academicScore = calculateAcademicScore(currentUser, candidate);
                double activityScore = candidate.getActivityScore() != null ? candidate.getActivityScore() : 0.0;
                double recencyScore = 0.5; // Default

                // Hybrid score (without content similarity for fallback)
                double relevanceScore = 
                    mutualFriendsScore * (WEIGHT_MUTUAL_FRIENDS + WEIGHT_CONTENT_SIMILARITY * 0.5) +
                    academicScore * (WEIGHT_ACADEMIC + WEIGHT_CONTENT_SIMILARITY * 0.5) +
                    activityScore * WEIGHT_ACTIVITY +
                    recencyScore * WEIGHT_RECENCY;

                String suggestionType = determineSuggestionType(mutualFriendsScore, academicScore);
                String suggestionReason = generateSuggestionReason(candidate, currentUser);

                return FriendRecommendationResponse.FriendSuggestion.builder()
                    .userId(candidate.getUserId())
                    .username(candidate.getUsername())
                    .fullName(candidate.getFullName())
                    .avatarUrl(candidate.getAvatarUrl())
                    .bio(candidate.getBio())
                    .facultyName(candidate.getFacultyName())
                    .majorName(candidate.getMajorName())
                    .batchYear(candidate.getBatchYear() != null ? candidate.getBatchYear().toString() : null)
                    .sameFaculty(candidate.isSameFaculty())
                    .sameMajor(candidate.isSameMajor())
                    .sameBatch(candidate.isSameBatch())
                    .mutualFriendsCount(candidate.getMutualFriendsCount())
                    .relevanceScore(relevanceScore)
                    .contentSimilarity(0.0) // Not available in fallback
                    .mutualFriendsScore(mutualFriendsScore)
                    .academicScore(academicScore)
                    .activityScore(activityScore)
                    .suggestionType(suggestionType)
                    .suggestionReason(suggestionReason)
                    .build();
            })
            .sorted((a, b) -> Double.compare(b.getRelevanceScore(), a.getRelevanceScore()))
            .limit(limit)
            .collect(Collectors.toList());
    }

    /**
     * Calculate additional scores for Python service
     */
    private Map<String, FriendRankingRequest.AdditionalScores> calculateAdditionalScores(
            String userId,
            UserAcademicProfile currentUser,
            List<FriendCandidateDTO> candidates) {

        Map<String, FriendRankingRequest.AdditionalScores> scores = new HashMap<>();

        // Get activity scores in batch
        List<String> candidateIds = candidates.stream()
            .map(FriendCandidateDTO::getUserId)
            .collect(Collectors.toList());
        
        Map<String, Double> activityScores = getActivityScoresBatch(candidateIds);

        for (FriendCandidateDTO candidate : candidates) {
            double mutualScore = calculateMutualFriendsScore(candidate.getMutualFriendsCount());
            double academicScore = calculateAcademicScore(currentUser, candidate);
            double activityScore = activityScores.getOrDefault(candidate.getUserId(), 0.0);
            double recencyScore = calculateRecencyScore(candidate);

            scores.put(candidate.getUserId(), FriendRankingRequest.AdditionalScores.builder()
                .mutualFriendsScore(mutualScore)
                .academicScore(academicScore)
                .activityScore(activityScore)
                .recencyScore(recencyScore)
                .mutualFriendsCount(candidate.getMutualFriendsCount())
                .build());
        }

        return scores;
    }

    /**
     * Calculate mutual friends score (normalized 0-1)
     */
    private double calculateMutualFriendsScore(int mutualCount) {
        // Score increases with mutual friends, capped at 10
        return Math.min(1.0, mutualCount / 10.0);
    }

    /**
     * Calculate academic connection score
     */
    private double calculateAcademicScore(UserAcademicProfile currentUser, FriendCandidateDTO candidate) {
        double score = 0.0;
        
        // Same faculty: +0.4
        if (currentUser.getFaculty() != null && 
            currentUser.getFaculty().equals(candidate.getFacultyName())) {
            score += 0.4;
        }
        
        // Same major: +0.4
        if (currentUser.getMajor() != null && 
            currentUser.getMajor().equals(candidate.getMajorName())) {
            score += 0.4;
        }
        
        // Same batch: +0.2
        if (currentUser.getBatch() != null && 
            currentUser.getBatch().equals(candidate.getBatchYear())) {
            score += 0.2;
        }
        
        return score;
    }

    /**
     * Calculate recency score based on user activity
     */
    private double calculateRecencyScore(FriendCandidateDTO candidate) {
        // If we have activity data, use it
        if (candidate.getActivityScore() != null && candidate.getActivityScore() > 0) {
            return candidate.getActivityScore();
        }
        return 0.5; // Default
    }

    /**
     * Get activity scores for multiple users
     */
    private Map<String, Double> getActivityScoresBatch(List<String> userIds) {
        Map<String, Double> scores = new HashMap<>();
        
        try {
            List<UserActivityScore> activityScores = activityScoreRepository.findByUserIdIn(userIds);
            for (UserActivityScore score : activityScores) {
                scores.put(score.getUserId(), score.getActivityScore());
            }
        } catch (Exception e) {
            log.warn("Failed to get batch activity scores: {}", e.getMessage());
        }
        
        return scores;
    }

    /**
     * Determine suggestion type based on scores
     */
    private String determineSuggestionType(double mutualScore, double academicScore) {
        if (mutualScore >= academicScore && mutualScore > 0.3) {
            return "MUTUAL_FRIENDS";
        } else if (academicScore > 0.5) {
            return "ACADEMIC_CONNECTION";
        } else if (mutualScore > 0) {
            return "FRIENDS_OF_FRIENDS";
        }
        return "ACTIVITY_BASED";
    }

    /**
     * Generate human-readable suggestion reason
     */
    private String generateSuggestionReason(FriendCandidateDTO candidate, UserAcademicProfile currentUser) {
        List<String> reasons = new ArrayList<>();

        if (candidate.getMutualFriendsCount() > 0) {
            reasons.add(candidate.getMutualFriendsCount() + " bạn chung");
        }

        if (candidate.isSameMajor() && candidate.getMajorName() != null) {
            reasons.add("Cùng ngành " + candidate.getMajorName());
        } else if (candidate.isSameFaculty() && candidate.getFacultyName() != null) {
            reasons.add("Cùng khoa " + candidate.getFacultyName());
        }

        if (candidate.isSameBatch() && candidate.getBatchYear() != null) {
            reasons.add("Cùng khóa " + candidate.getBatchYear());
        }

        return reasons.isEmpty() ? "Gợi ý cho bạn" : String.join(" • ", reasons);
    }

    /**
     * Get excluded user IDs (recently shown, dismissed)
     */
    private Set<String> getExcludedUserIds(String userId) {
        Set<String> excluded = new HashSet<>();
        
        try {
            LocalDateTime since = LocalDateTime.now().minusHours(24);
            excluded.addAll(logRepository.findRecentlyRecommendedUserIds(userId, since));
            excluded.addAll(logRepository.findDismissedUserIds(userId));
        } catch (Exception e) {
            log.warn("Failed to get excluded user IDs: {}", e.getMessage());
        }
        
        return excluded;
    }

    /**
     * Cache suggestions
     */
    private void cacheSuggestions(String userId, List<FriendRecommendationResponse.FriendSuggestion> suggestions) {
        try {
            redisCacheService.cacheRecommendations(
                FRIEND_SUGGESTION_CACHE_KEY + userId,
                suggestions,
                Duration.ofHours(cacheTtlHours),
                true // Use raw key
            );
        } catch (Exception e) {
            log.warn("Failed to cache friend suggestions: {}", e.getMessage());
        }
    }

    /**
     * Get cached suggestions
     */
    @SuppressWarnings("unchecked")
    private List<FriendRecommendationResponse.FriendSuggestion> getCachedSuggestions(String userId) {
        try {
            return (List<FriendRecommendationResponse.FriendSuggestion>) 
                redisCacheService.getRecommendations(
                    FRIEND_SUGGESTION_CACHE_KEY + userId,
                    FriendRecommendationResponse.FriendSuggestion.class,
                    true // Use raw key
                );
        } catch (Exception e) {
            log.warn("Failed to get cached suggestions: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Log recommendations for analytics
     */
    @Transactional
    public void logRecommendations(String userId, List<FriendRecommendationResponse.FriendSuggestion> suggestions) {
        try {
            List<FriendRecommendationLog> logs = new ArrayList<>();
            int rank = 1;
            
            for (FriendRecommendationResponse.FriendSuggestion suggestion : suggestions) {
                FriendRecommendationLog log = FriendRecommendationLog.builder()
                    .userId(userId)
                    .recommendedUserId(suggestion.getUserId())
                    .relevanceScore((float) suggestion.getRelevanceScore())
                    .contentSimilarity((float) suggestion.getContentSimilarity())
                    .mutualFriendsScore((float) suggestion.getMutualFriendsScore())
                    .academicScore((float) suggestion.getAcademicScore())
                    .activityScore((float) suggestion.getActivityScore())
                    .suggestionType(FriendRecommendationLog.SuggestionType.valueOf(suggestion.getSuggestionType()))
                    .suggestionReason(suggestion.getSuggestionReason())
                    .rankPosition(rank++)
                    .build();
                logs.add(log);
            }
            
            logRepository.saveAll(logs);
        } catch (Exception e) {
            log.warn("Failed to log recommendations: {}", e.getMessage());
        }
    }

    /**
     * Record user feedback on a suggestion
     */
    @Transactional
    public void recordFeedback(String userId, String recommendedUserId, String action) {
        try {
            List<FriendRecommendationLog> logs = logRepository.findByUserIdAndRecommendedUserId(userId, recommendedUserId);
            
            if (!logs.isEmpty()) {
                FriendRecommendationLog log = logs.get(0);
                LocalDateTime now = LocalDateTime.now();
                
                switch (action.toUpperCase()) {
                    case "CLICK" -> log.setClickedAt(now);
                    case "REQUEST" -> log.setFriendRequestSentAt(now);
                    case "ACCEPT" -> log.setAcceptedAt(now);
                    case "REJECT" -> log.setRejectedAt(now);
                    case "DISMISS" -> log.setDismissedAt(now);
                }
                
                logRepository.save(log);
            }
        } catch (Exception e) {
            log.warn("Failed to record feedback: {}", e.getMessage());
        }
    }

    /**
     * Invalidate cache for a user
     */
    public void invalidateCache(String userId) {
        try {
            redisCacheService.invalidateRecommendations(FRIEND_SUGGESTION_CACHE_KEY + userId);
        } catch (Exception e) {
            log.warn("Failed to invalidate cache: {}", e.getMessage());
        }
    }

    /**
     * Build response object
     */
    private FriendRecommendationResponse buildResponse(
            String userId,
            List<FriendRecommendationResponse.FriendSuggestion> suggestions,
            int limit,
            long startTime,
            String source) {

        // Add rank positions
        List<FriendRecommendationResponse.FriendSuggestion> rankedSuggestions = new ArrayList<>();
        int rank = 1;
        for (FriendRecommendationResponse.FriendSuggestion s : suggestions.stream().limit(limit).toList()) {
            s.setRankPosition(rank++);
            rankedSuggestions.add(s);
        }

        return FriendRecommendationResponse.builder()
            .userId(userId)
            .suggestions(rankedSuggestions)
            .count(rankedSuggestions.size())
            .metadata(FriendRecommendationResponse.ResponseMetadata.builder()
                .source(source)
                .processingTimeMs(System.currentTimeMillis() - startTime)
                .timestamp(LocalDateTime.now())
                .modelVersion(pythonServiceEnabled ? "phobert-v1" : "rule-based")
                .mlEnabled(pythonServiceEnabled)
                .build())
            .build();
    }
}
