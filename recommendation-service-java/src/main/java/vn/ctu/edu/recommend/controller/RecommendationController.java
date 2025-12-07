package vn.ctu.edu.recommend.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import vn.ctu.edu.recommend.model.dto.FeedbackRequest;
import vn.ctu.edu.recommend.model.dto.RecommendationRequest;
import vn.ctu.edu.recommend.model.dto.RecommendationResponse;
import vn.ctu.edu.recommend.service.RecommendationService;

import java.time.LocalDateTime;
import java.util.Map;

/**
 * REST controller for recommendation endpoints
 */
@RestController
@RequestMapping("/api/recommend")
@Slf4j
@RequiredArgsConstructor
public class RecommendationController {

    private final RecommendationService recommendationService;

    /**
     * GET /api/recommend/posts?userId={id}&page=0&size=20
     * Get personalized post recommendations
     */
    @GetMapping("/posts")
    public ResponseEntity<RecommendationResponse> getRecommendedPosts(
            @RequestParam String userId,
            @RequestParam(defaultValue = "0") Integer page,
            @RequestParam(defaultValue = "20") Integer size,
            @RequestParam(required = false) Boolean includeExplanations) {
        
        log.info("GET /api/recommend/posts - userId: {}, page: {}, size: {}", userId, page, size);

        RecommendationRequest request = RecommendationRequest.builder()
            .userId(userId)
            .page(page)
            .size(size)
            .includeExplanations(includeExplanations != null ? includeExplanations : false)
            .build();

        RecommendationResponse response = recommendationService.getRecommendations(request);
        return ResponseEntity.ok(response);
    }

    /**
     * POST /api/recommend/posts
     * Get recommendations with advanced options
     */
    @PostMapping("/posts")
    public ResponseEntity<RecommendationResponse> getRecommendations(
            @Valid @RequestBody RecommendationRequest request) {
        
        log.info("POST /api/recommend/posts - userId: {}", request.getUserId());

        RecommendationResponse response = recommendationService.getRecommendations(request);
        return ResponseEntity.ok(response);
    }

    /**
     * POST /api/recommend/feedback
     * Record user feedback
     */
    @PostMapping("/feedback")
    public ResponseEntity<Map<String, Object>> recordFeedback(
            @Valid @RequestBody FeedbackRequest request) {
        
        log.info("POST /api/recommend/feedback - userId: {}, postId: {}, type: {}", 
            request.getUserId(), request.getPostId(), request.getFeedbackType());

        recommendationService.recordFeedback(request);

        return ResponseEntity.ok(Map.of(
            "success", true,
            "message", "Feedback recorded successfully",
            "timestamp", LocalDateTime.now()
        ));
    }

    /**
     * POST /api/recommend/embedding/rebuild
     * Trigger embedding rebuild (admin endpoint)
     */
    @PostMapping("/embedding/rebuild")
    public ResponseEntity<Map<String, Object>> rebuildEmbeddings() {
        log.info("POST /api/recommend/embedding/rebuild - Triggered by admin");

        recommendationService.rebuildEmbeddings();

        return ResponseEntity.ok(Map.of(
            "success", true,
            "message", "Embedding rebuild started",
            "timestamp", LocalDateTime.now()
        ));
    }

    /**
     * POST /api/recommend/rank/rebuild
     * Trigger recommendation cache rebuild (admin endpoint)
     */
    @PostMapping("/rank/rebuild")
    public ResponseEntity<Map<String, Object>> rebuildRankings() {
        log.info("POST /api/recommend/rank/rebuild - Triggered by admin");

        recommendationService.rebuildRecommendationCache();

        return ResponseEntity.ok(Map.of(
            "success", true,
            "message", "Recommendation cache rebuilt",
            "timestamp", LocalDateTime.now()
        ));
    }

    /**
     * DELETE /api/recommend/cache/{userId}
     * Invalidate cache for specific user
     */
    @DeleteMapping("/cache/{userId}")
    public ResponseEntity<Map<String, Object>> invalidateUserCache(
            @PathVariable String userId) {
        
        log.info("DELETE /api/recommend/cache/{} - Invalidating cache", userId);

        recommendationService.invalidateUserCache(userId);

        return ResponseEntity.ok(Map.of(
            "success", true,
            "message", "Cache invalidated for user: " + userId,
            "timestamp", LocalDateTime.now()
        ));
    }

    /**
     * GET /api/recommend/health
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        return ResponseEntity.ok(Map.of(
            "status", "UP",
            "service", "recommendation-service",
            "timestamp", LocalDateTime.now()
        ));
    }
}
