package vn.ctu.edu.recommend.client;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import vn.ctu.edu.recommend.model.dto.PythonModelRequest;
import vn.ctu.edu.recommend.model.dto.PythonModelResponse;

import java.time.Duration;

/**
 * Client for communicating with Python ML Model Service
 * Handles POST /predict endpoint for ranking recommendations
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class PythonModelServiceClient {

    private final WebClient.Builder webClientBuilder;

    @Value("${recommendation.python-service.url:http://localhost:8096}")
    private String pythonServiceUrl;

    @Value("${recommendation.python-service.predict-endpoint:/api/model/predict}")
    private String predictEndpoint;

    @Value("${recommendation.python-service.timeout:10000}")
    private long timeout;

    /**
     * Call Python model service to rank posts
     * @param request contains user profile, history, and candidate posts
     * @return ranked posts with scores
     */
    public PythonModelResponse predictRanking(PythonModelRequest request) {
        try {
            log.info("🚀 Calling Python model service for user: {}", request.getUserAcademic().getUserId());
            log.debug("   User academic: major={}, faculty={}", 
                request.getUserAcademic().getMajor(), 
                request.getUserAcademic().getFaculty());
            log.debug("   User history size: {}", request.getUserHistory().size());
            log.debug("   Candidate posts: {}", request.getCandidatePosts().size());
            log.debug("   TopK requested: {}", request.getTopK());
            
            // Log first candidate post for debugging
            if (!request.getCandidatePosts().isEmpty()) {
                var firstPost = request.getCandidatePosts().get(0);
                log.debug("   Sample post: id={}, contentLength={}, likes={}, comments={}, shares={}",
                    firstPost.getPostId(),
                    firstPost.getContent() != null ? firstPost.getContent().length() : 0,
                    firstPost.getLikeCount(),
                    firstPost.getCommentCount(),
                    firstPost.getShareCount());
            }

            WebClient webClient = webClientBuilder.baseUrl(pythonServiceUrl).build();

            PythonModelResponse response = webClient.post()
                .uri(predictEndpoint)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PythonModelResponse.class)
                .timeout(Duration.ofMillis(timeout))
                .onErrorResume(e -> {
                    log.error("❌ Python model service error: {} - {}", e.getClass().getSimpleName(), e.getMessage());
                    if (e.getMessage() != null && e.getMessage().contains("422")) {
                        log.error("   422 Unprocessable Entity - Request validation failed");
                        log.error("   Check if request fields match Python PredictionRequest schema");
                    }
                    return Mono.just(createFallbackResponse());
                })
                .block();

            if (response != null && response.getRankedPosts() != null) {
                log.info("✅ Received {} ranked posts from Python service", response.getRankedPosts().size());
                return response;
            }

            log.warn("⚠️  Received empty response from Python service, using fallback");
            return createFallbackResponse();

        } catch (Exception e) {
            log.error("❌ Error calling Python model service: {}", e.getMessage(), e);
            return createFallbackResponse();
        }
    }

    /**
     * Health check for Python model service
     */
    public boolean isServiceHealthy() {
        try {
            WebClient webClient = webClientBuilder.baseUrl(pythonServiceUrl).build();
            
            String response = webClient.get()
                .uri("/health")
                .retrieve()
                .bodyToMono(String.class)
                .timeout(Duration.ofMillis(5000))
                .block();

            return response != null && response.contains("healthy");
        } catch (Exception e) {
            log.warn("Python model service health check failed: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Create fallback response when Python service is unavailable
     * Returns empty ranked list
     */
    private PythonModelResponse createFallbackResponse() {
        return PythonModelResponse.builder()
            .rankedPosts(java.util.Collections.emptyList())
            .modelVersion("fallback")
            .processingTimeMs(0L)
            .build();
    }
}
