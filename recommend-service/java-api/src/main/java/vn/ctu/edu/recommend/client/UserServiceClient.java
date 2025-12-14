package vn.ctu.edu.recommend.client;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import vn.ctu.edu.recommend.model.dto.FriendCandidateDTO;
import vn.ctu.edu.recommend.model.dto.UserAcademicProfile;

import java.time.Duration;
import java.util.Collections;
import java.util.List;

/**
 * Client for fetching user data from user-service
 */
@Component
@Slf4j
public class UserServiceClient {

    private final WebClient.Builder webClientBuilder;
    
    @Value("${user-service.url:http://localhost:8081}")
    private String userServiceUrl;

    @Value("${user-service.timeout:5000}")
    private long timeout;

    public UserServiceClient(WebClient.Builder webClientBuilder) {
        this.webClientBuilder = webClientBuilder;
    }

    /**
     * Fetch user academic profile from user-service
     */
    public UserAcademicProfile getUserAcademicProfile(String userId) {
        try {
            log.debug("Fetching academic profile for user: {} from {}", userId, userServiceUrl);
            
            WebClient webClient = webClientBuilder.baseUrl(userServiceUrl).build();
            
            UserAcademicProfile profile = webClient.get()
                .uri("/api/users/{userId}/academic-profile", userId)
                .retrieve()
                .bodyToMono(UserAcademicProfile.class)
                .timeout(Duration.ofMillis(timeout))
                .onErrorResume(e -> {
                    log.warn("Failed to fetch user profile: {}", e.getMessage());
                    return Mono.just(createFallbackProfile(userId));
                })
                .block();

            return profile != null ? profile : createFallbackProfile(userId);

        } catch (Exception e) {
            log.error("Error fetching user profile: {}", e.getMessage(), e);
            return createFallbackProfile(userId);
        }
    }

    // ==================== Friend Recommendation Methods ====================

    /**
     * Get friend candidates for a user
     * Returns users who:
     * - Are not already friends
     * - Have not blocked/been blocked
     * - Don't have pending friend request
     * - Prioritized by faculty/major match
     */
    public List<FriendCandidateDTO> getFriendCandidates(String userId, int limit) {
        try {
            log.debug("Fetching friend candidates for user: {}, limit: {} from {}", userId, limit, userServiceUrl);
            
            WebClient webClient = webClientBuilder.baseUrl(userServiceUrl).build();
            
            List<FriendCandidateDTO> candidates = webClient.get()
                .uri(uriBuilder -> uriBuilder
                    .path("/api/users/{userId}/friend-candidates")
                    .queryParam("limit", limit)
                    .build(userId))
                .retrieve()
                .bodyToFlux(FriendCandidateDTO.class)
                .timeout(Duration.ofMillis(timeout))
                .collectList()
                .onErrorResume(e -> {
                    log.warn("Failed to fetch friend candidates: {}", e.getMessage());
                    return Mono.just(Collections.emptyList());
                })
                .block();

            return candidates != null ? candidates : Collections.emptyList();

        } catch (Exception e) {
            log.error("Error fetching friend candidates: {}", e.getMessage(), e);
            return Collections.emptyList();
        }
    }

    /**
     * Get mutual friends count between two users
     */
    public int getMutualFriendsCount(String userId1, String userId2) {
        try {
            log.debug("Getting mutual friends count: {} <-> {}", userId1, userId2);
            
            WebClient webClient = webClientBuilder.baseUrl(userServiceUrl).build();
            
            Integer count = webClient.get()
                .uri("/api/users/mutual-count/{userId1}/{userId2}", userId1, userId2)
                .retrieve()
                .bodyToMono(Integer.class)
                .timeout(Duration.ofMillis(timeout))
                .onErrorResume(e -> {
                    log.warn("Failed to get mutual friends count: {}", e.getMessage());
                    return Mono.just(0);
                })
                .block();

            return count != null ? count : 0;

        } catch (Exception e) {
            log.error("Error getting mutual friends count: {}", e.getMessage());
            return 0;
        }
    }

    /**
     * Get full user profile by ID
     */
    public FriendCandidateDTO getUserProfile(String userId) {
        try {
            log.debug("Fetching full user profile: {}", userId);
            
            WebClient webClient = webClientBuilder.baseUrl(userServiceUrl).build();
            
            return webClient.get()
                .uri("/api/users/{userId}/profile", userId)
                .retrieve()
                .bodyToMono(FriendCandidateDTO.class)
                .timeout(Duration.ofMillis(timeout))
                .onErrorResume(e -> {
                    log.warn("Failed to fetch user profile: {}", e.getMessage());
                    return Mono.empty();
                })
                .block();

        } catch (Exception e) {
            log.error("Error fetching user profile: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Batch get mutual friends count for multiple user pairs
     */
    public java.util.Map<String, Integer> getMutualFriendsCountBatch(String userId, List<String> targetUserIds) {
        java.util.Map<String, Integer> result = new java.util.HashMap<>();
        
        // For now, fetch individually (could be optimized with batch API)
        for (String targetId : targetUserIds) {
            int count = getMutualFriendsCount(userId, targetId);
            result.put(targetId, count);
        }
        
        return result;
    }

    /**
     * Check if two users are already friends
     */
    public boolean areFriends(String userId1, String userId2) {
        try {
            WebClient webClient = webClientBuilder.baseUrl(userServiceUrl).build();
            
            Boolean result = webClient.get()
                .uri("/api/users/{userId1}/is-friend/{userId2}", userId1, userId2)
                .retrieve()
                .bodyToMono(Boolean.class)
                .timeout(Duration.ofMillis(timeout))
                .onErrorResume(e -> Mono.just(false))
                .block();

            return result != null && result;

        } catch (Exception e) {
            log.error("Error checking friendship: {}", e.getMessage());
            return false;
        }
    }

    private UserAcademicProfile createFallbackProfile(String userId) {
        return UserAcademicProfile.builder()
            .userId(userId)
            .major("unknown")
            .faculty("unknown")
            .degree("unknown")
            .batch("unknown")
            .build();
    }
}

