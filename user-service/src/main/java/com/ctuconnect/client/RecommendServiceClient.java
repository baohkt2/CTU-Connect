package com.ctuconnect.client;

import com.ctuconnect.dto.FriendSuggestionDTO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Client for communicating with Recommend Service
 * Retrieves ML-enhanced friend suggestions
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class RecommendServiceClient {

    private final WebClient.Builder webClientBuilder;

    @Value("${recommend-service.url:http://localhost:8095}")
    private String recommendServiceUrl;

    @Value("${recommend-service.enabled:true}")
    private boolean recommendServiceEnabled;

    @Value("${recommend-service.timeout-ms:5000}")
    private int timeoutMs;

    /**
     * Get ML-enhanced friend suggestions from Recommend Service
     */
    public List<FriendSuggestionDTO> getMLFriendSuggestions(String userId, int limit) {
        if (!recommendServiceEnabled) {
            log.debug("Recommend service disabled, skipping ML suggestions");
            return Collections.emptyList();
        }

        try {
            log.info("🤖 Fetching ML friend suggestions for user: {}", userId);

            FriendRecommendationResponse response = webClientBuilder.build()
                .get()
                .uri(recommendServiceUrl + "/api/recommendations/friends/{userId}?limit={limit}",
                    userId, limit)
                .retrieve()
                .onStatus(status -> status.isError(), clientResponse -> {
                    log.warn("Recommend service returned error: {}", clientResponse.statusCode());
                    return Mono.empty();
                })
                .bodyToMono(FriendRecommendationResponse.class)
                .timeout(Duration.ofMillis(timeoutMs))
                .onErrorResume(ex -> {
                    log.warn("Failed to fetch ML suggestions: {}", ex.getMessage());
                    return Mono.empty();
                })
                .block();

            // Debug logging
            if (response != null) {
                log.debug("🔍 Response received - userId: {}, count: {}, suggestions: {}", 
                    response.getUserId(), response.getCount(), 
                    response.getSuggestions() != null ? response.getSuggestions().size() : "null");
            } else {
                log.warn("⚠️ Response is null from recommend-service");
            }

            if (response != null && response.getSuggestions() != null && !response.getSuggestions().isEmpty()) {
                log.info("✅ Received {} ML suggestions from recommend-service", response.getSuggestions().size());
                return convertToFriendSuggestions(response.getSuggestions());
            }

            log.warn("⚠️ No suggestions in response (response null: {}, suggestions null: {})", 
                response == null, response != null && response.getSuggestions() == null);
            return Collections.emptyList();

        } catch (Exception e) {
            log.error("Error calling recommend service: {}", e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * Record feedback for a friend suggestion
     */
    public void recordFeedback(String userId, String recommendedUserId, String action) {
        if (!recommendServiceEnabled) {
            return;
        }

        try {
            webClientBuilder.build()
                .post()
                .uri(recommendServiceUrl + "/api/recommendations/friends/{userId}/feedback", userId)
                .bodyValue(Map.of(
                    "recommendedUserId", recommendedUserId,
                    "action", action
                ))
                .retrieve()
                .bodyToMono(Void.class)
                .timeout(Duration.ofMillis(timeoutMs))
                .subscribe(
                    success -> log.debug("Feedback recorded successfully"),
                    error -> log.warn("Failed to record feedback: {}", error.getMessage())
                );
        } catch (Exception e) {
            log.warn("Error recording feedback: {}", e.getMessage());
        }
    }

    /**
     * Invalidate friend suggestions cache for a user
     */
    public void invalidateCache(String userId) {
        if (!recommendServiceEnabled) {
            return;
        }

        try {
            webClientBuilder.build()
                .delete()
                .uri(recommendServiceUrl + "/api/recommendations/friends/{userId}/cache", userId)
                .retrieve()
                .bodyToMono(Void.class)
                .timeout(Duration.ofMillis(timeoutMs))
                .subscribe(
                    success -> log.debug("Cache invalidated successfully"),
                    error -> log.warn("Failed to invalidate cache: {}", error.getMessage())
                );
        } catch (Exception e) {
            log.warn("Error invalidating cache: {}", e.getMessage());
        }
    }

    /**
     * Convert ML suggestions to FriendSuggestionDTO
     */
    private List<FriendSuggestionDTO> convertToFriendSuggestions(List<MLFriendSuggestion> mlSuggestions) {
        return mlSuggestions.stream()
            .map(ml -> FriendSuggestionDTO.builder()
                .userId(ml.getUserId())
                .id(ml.getUserId())  // Alias for frontend compatibility
                .username(ml.getUsername())
                .fullName(ml.getFullName())
                .avatarUrl(ml.getAvatarUrl())
                .bio(ml.getBio())
                .facultyName(ml.getFacultyName())
                .majorName(ml.getMajorName())
                .batchYear(ml.getBatchYear()) // Already String, no conversion needed
                .sameFaculty(ml.isSameFaculty())
                .sameMajor(ml.isSameMajor())
                .sameBatch(ml.isSameBatch())
                .mutualFriendsCount(ml.getMutualFriendsCount())
                .relevanceScore(ml.getRelevanceScore())
                .suggestionReason(ml.getSuggestionReason())
                .suggestionType(mapSuggestionType(ml.getSuggestionType()))
                .contentSimilarity(ml.getContentSimilarity())
                .academicScore(ml.getAcademicScore())
                .activityScore(ml.getActivityScore())
                .mlEnhanced(true)
                .build())
            .toList();
    }

    private FriendSuggestionDTO.SuggestionType mapSuggestionType(String type) {
        if (type == null) {
            return FriendSuggestionDTO.SuggestionType.ACTIVITY_BASED;
        }
        
        return switch (type.toUpperCase()) {
            case "MUTUAL_FRIENDS" -> FriendSuggestionDTO.SuggestionType.MUTUAL_FRIENDS;
            case "ACADEMIC_CONNECTION" -> FriendSuggestionDTO.SuggestionType.ACADEMIC_CONNECTION;
            case "FRIENDS_OF_FRIENDS" -> FriendSuggestionDTO.SuggestionType.FRIENDS_OF_FRIENDS;
            case "CONTENT_SIMILARITY" -> FriendSuggestionDTO.SuggestionType.CONTENT_SIMILARITY;
            case "PROFILE_VIEWERS" -> FriendSuggestionDTO.SuggestionType.PROFILE_VIEWERS;
            default -> FriendSuggestionDTO.SuggestionType.ACTIVITY_BASED;
        };
    }

    /**
     * Internal response class for Recommend Service
     */
    @lombok.Data
    public static class FriendRecommendationResponse {
        private String userId;
        private List<MLFriendSuggestion> suggestions;
        private int count;
        private ResponseMetadata metadata;

        @lombok.Data
        public static class ResponseMetadata {
            private String source;
            private Long processingTimeMs;
            private String modelVersion;
            private boolean mlEnabled;
        }
    }

    /**
     * Internal DTO for ML suggestions from Recommend Service
     * Must match FriendRecommendationResponse.FriendSuggestion from recommend-service
     */
    @lombok.Data
    public static class MLFriendSuggestion {
        private String userId;
        private String username;
        private String fullName;
        private String avatarUrl;
        private String bio;
        private String facultyName;
        private String majorName;
        private String batchYear; // Changed from Integer to String to match recommend-service
        private boolean sameFaculty;
        private boolean sameMajor;
        private boolean sameBatch;
        private int mutualFriendsCount;
        private double relevanceScore;
        private double contentSimilarity;
        private double mutualFriendsScore;
        private double academicScore;
        private double activityScore;
        private String suggestionType;
        private String suggestionReason;
        private int rankPosition;
    }
}
