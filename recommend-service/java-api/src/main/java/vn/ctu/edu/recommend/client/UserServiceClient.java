package vn.ctu.edu.recommend.client;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import vn.ctu.edu.recommend.model.dto.UserAcademicProfile;

import java.time.Duration;

/**
 * Client for fetching user academic profile from user-service
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class UserServiceClient {

    private final WebClient.Builder webClientBuilder;

    /**
     * Fetch user academic profile from user-service via service discovery
     */
    public UserAcademicProfile getUserAcademicProfile(String userId) {
        try {
            log.debug("Fetching academic profile for user: {}", userId);
            
            WebClient webClient = webClientBuilder.baseUrl("http://user-service").build();
            
            UserAcademicProfile profile = webClient.get()
                .uri("/api/users/{userId}/academic-profile", userId)
                .retrieve()
                .bodyToMono(UserAcademicProfile.class)
                .timeout(Duration.ofMillis(5000))
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
