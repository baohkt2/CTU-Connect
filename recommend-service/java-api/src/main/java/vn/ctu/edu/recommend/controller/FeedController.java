package vn.ctu.edu.recommend.controller;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import vn.ctu.edu.recommend.model.dto.RecommendationResponse;
import vn.ctu.edu.recommend.model.enums.FeedbackType;
import vn.ctu.edu.recommend.service.HybridRecommendationService;

import java.util.Map;

/**
 * Controller for hybrid recommendation feed
 * Main API endpoint: GET /api/recommendation/feed
 */
@RestController
@RequestMapping("/api/recommendation")
@RequiredArgsConstructor
@Slf4j
public class FeedController {

    private final HybridRecommendationService hybridRecommendationService;

    /**
     * Get personalized feed for user
     * Uses hybrid architecture: Java orchestration + Python ML model
     * 
     * @param userId User ID
     * @param page Page number (default 0)
     * @param size Number of items per page (default 20)
     * @return Personalized feed recommendations
     */
    @GetMapping("/feed")
    public ResponseEntity<RecommendationResponse> getFeed(
            @RequestParam String userId,
            @RequestParam(required = false, defaultValue = "0") Integer page,
            @RequestParam(required = false, defaultValue = "20") Integer size) {
        
        log.info("Feed request: userId={}, page={}, size={}", userId, page, size);
        
        RecommendationResponse response = hybridRecommendationService.getFeed(userId, page, size);
        
        return ResponseEntity.ok(response);
    }

    /**
     * Record user interaction with a post
     * Sends event to Kafka for Python training pipeline
     * 
     * @param request Interaction details
     */
    @PostMapping("/interaction")
    public ResponseEntity<Map<String, String>> recordInteraction(@RequestBody InteractionRequest request) {
        
        log.info("Interaction: userId={}, postId={}, type={}", 
            request.getUserId(), request.getPostId(), request.getType());
        
        FeedbackType feedbackType = FeedbackType.valueOf(request.getType().toUpperCase());
        
        hybridRecommendationService.recordInteraction(
            request.getUserId(), 
            request.getPostId(), 
            feedbackType,
            request.getViewDuration(),
            request.getContext()
        );
        
        return ResponseEntity.ok(Map.of(
            "status", "success",
            "message", "Interaction recorded"
        ));
    }

    /**
     * Invalidate user's recommendation cache
     * Useful when user profile changes or for testing
     */
    @PostMapping("/cache/invalidate")
    public ResponseEntity<Map<String, String>> invalidateCache(@RequestParam String userId) {
        log.info("Invalidating cache for user: {}", userId);
        // Implemented in Redis cache service
        return ResponseEntity.ok(Map.of(
            "status", "success",
            "message", "Cache invalidated for user: " + userId
        ));
    }

    /**
     * Health check endpoint for Python model service
     */
    @GetMapping("/health/python-service")
    public ResponseEntity<Map<String, Object>> checkPythonServiceHealth() {
        // Implementation would check Python service health
        return ResponseEntity.ok(Map.of(
            "status", "checking",
            "message", "Python model service health check"
        ));
    }

    /**
     * DTO for interaction request
     */
    @Data
    public static class InteractionRequest {
        private String userId;
        private String postId;
        private String type;  // VIEW, LIKE, COMMENT, SHARE
        private Double viewDuration;
        private Map<String, Object> context;
    }
}
