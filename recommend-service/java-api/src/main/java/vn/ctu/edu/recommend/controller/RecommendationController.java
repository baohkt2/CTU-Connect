package vn.ctu.edu.recommend.controller;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import vn.ctu.edu.recommend.model.dto.RecommendationResponse;
import vn.ctu.edu.recommend.model.enums.FeedbackType;
import vn.ctu.edu.recommend.service.HybridRecommendationService;

import java.time.LocalDateTime;
import java.util.*;

/**
 * Unified REST Controller for Recommendation Service
 * Following ARCHITECTURE.md - Java as Orchestrator
 * 
 * Main endpoints:
 * - GET /api/recommendations/feed - Get personalized feed
 * - POST /api/recommendations/interaction - Record user interaction
 * - POST /api/recommendations/refresh - Refresh user cache
 * - GET /api/recommendations/health - Health check
 */
@RestController
@RequestMapping("/api/recommendations")
@RequiredArgsConstructor
@Slf4j
public class RecommendationController {

    private final HybridRecommendationService recommendationService;

    /**
     * GET /api/recommendations/feed
     * Main endpoint - Get personalized feed for user
     * Uses hybrid architecture: Java orchestration + Python ML model
     * 
     * @param userId User ID
     * @param page Page number (default 0)
     * @param size Number of items per page (default 20)
     * @param excludePostIds Comma-separated list of post IDs to exclude (for pagination)
     * @return Personalized feed recommendations
     */
    @GetMapping("/feed")
    public ResponseEntity<RecommendationResponse> getFeed(
            @RequestParam String userId,
            @RequestParam(required = false, defaultValue = "0") Integer page,
            @RequestParam(required = false, defaultValue = "20") Integer size,
            @RequestParam(required = false) String excludePostIds) {
        
        // Parse excluded post IDs (CSV format: "id1,id2,id3")
        Set<String> excludedIds = new HashSet<>();
        if (excludePostIds != null && !excludePostIds.isEmpty()) {
            excludedIds.addAll(Arrays.asList(excludePostIds.split(",")));
        }
        
        log.info("========================================");
        log.info("📥 API REQUEST: GET /api/recommendations/feed");
        log.info("   User ID: {}", userId);
        log.info("   Page: {}, Size: {}, Exclude: {} posts", page, size, excludedIds.size());
        log.info("========================================");
        
        try {
            log.debug("🔄 Calling hybrid recommendation service for feed generation");
            RecommendationResponse response = recommendationService.getFeed(userId, page, size, excludedIds);
            
            log.info("========================================");
            log.info("📤 API RESPONSE: GET /api/recommendations/feed");
            log.info("   Total Items: {}", response.getRecommendations() != null ? response.getRecommendations().size() : 0);
            log.info("   User ID: {}", response.getUserId());
            
            // 🔍 DEBUG: Log detailed post list with scores
            if (response.getRecommendations() != null && !response.getRecommendations().isEmpty()) {
                log.info("📋 RECOMMENDED POSTS LIST:");
                log.info("   Format: [Rank] PostID -> Score");
                log.info("   ----------------------------------------");
                
                for (int i = 0; i < response.getRecommendations().size(); i++) {
                    var post = response.getRecommendations().get(i);
                    Double score = post.getScore();
                    
                    log.info("   [{}] {} -> score: {}", 
                        String.format("%2d", i + 1),
                        post.getPostId(),
                        String.format("%.4f", score != null ? score : 0.0)
                    );
                }
                
                log.info("   ----------------------------------------");
                log.info("📊 SCORE STATISTICS:");
                
                var scores = response.getRecommendations().stream()
                    .map(p -> p.getScore() != null ? p.getScore() : 0.0)
                    .collect(java.util.stream.Collectors.toList());
                
                double maxScore = scores.stream().max(Double::compare).orElse(0.0);
                double minScore = scores.stream().min(Double::compare).orElse(0.0);
                double avgScore = scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                
                log.info("   Max Score: {}", String.format("%.4f", maxScore));
                log.info("   Min Score: {}", String.format("%.4f", minScore));
                log.info("   Avg Score: {}", String.format("%.4f", avgScore));
            }
            
            log.info("========================================");
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("========================================");
            log.error("❌ ERROR: GET /api/recommendations/feed failed");
            log.error("   User ID: {}", userId);
            log.error("   Error: {}", e.getMessage(), e);
            log.error("========================================");
            throw e;
        }
    }

    /**
     * POST /api/recommendations/interaction
     * Record user interaction with a post
     * Sends event to Kafka for Python training pipeline
     * 
     * @param request Interaction details
     */
    @PostMapping("/interaction")
    public ResponseEntity<Map<String, String>> recordInteraction(@RequestBody InteractionRequest request) {
        
        log.info("========================================");
        log.info("📥 API REQUEST: POST /api/recommendations/interaction");
        log.info("   User ID: {}", request.getUserId());
        log.info("   Post ID: {}", request.getPostId());
        log.info("   Type: {}", request.getType());
        log.info("   View Duration: {}", request.getViewDuration());
        log.info("========================================");
        
        try {
            FeedbackType feedbackType = FeedbackType.valueOf(request.getType().toUpperCase());
            
            log.debug("🔄 Recording interaction in hybrid service");
            recommendationService.recordInteraction(
                request.getUserId(), 
                request.getPostId(), 
                feedbackType,
                request.getViewDuration(),
                request.getContext()
            );
            
            log.info("✅ Interaction recorded successfully");
            
            return ResponseEntity.ok(Map.of(
                "status", "success",
                "message", "Interaction recorded"
            ));
        } catch (Exception e) {
            log.error("❌ ERROR recording interaction: {}", e.getMessage(), e);
            return ResponseEntity.ok(Map.of(
                "status", "error",
                "message", "Failed to record interaction: " + e.getMessage()
            ));
        }
    }

    /**
     * POST /api/recommendations/refresh
     * Invalidate user's recommendation cache
     * Useful when user profile changes or for testing
     */
    @PostMapping("/refresh")
    public ResponseEntity<Map<String, String>> refreshCache(@RequestParam String userId) {
        log.info("🔄 Refreshing cache for user: {}", userId);
        
        try {
            recommendationService.invalidateUserCache(userId);
            
            return ResponseEntity.ok(Map.of(
                "status", "success",
                "message", "Cache refreshed for user: " + userId
            ));
        } catch (Exception e) {
            log.error("❌ ERROR refreshing cache: {}", e.getMessage(), e);
            return ResponseEntity.ok(Map.of(
                "status", "error",
                "message", "Failed to refresh cache: " + e.getMessage()
            ));
        }
    }

    /**
     * DELETE /api/recommendations/cache/{userId}
     * Invalidate cache for specific user (alternative endpoint)
     */
    @DeleteMapping("/cache/{userId}")
    public ResponseEntity<Map<String, Object>> invalidateCache(@PathVariable String userId) {
        log.info("🗑️  Invalidating cache for user: {}", userId);
        
        try {
            recommendationService.invalidateUserCache(userId);
            
            return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "Cache invalidated for user: " + userId,
                "timestamp", LocalDateTime.now()
            ));
        } catch (Exception e) {
            log.error("❌ ERROR invalidating cache: {}", e.getMessage(), e);
            throw e;
        }
    }

    /**
     * GET /api/recommendations/health
     * Health check endpoint for recommendation service
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        return ResponseEntity.ok(Map.of(
            "status", "UP",
            "service", "recommendation-service-java",
            "timestamp", LocalDateTime.now()
        ));
    }

    /**
     * GET /api/recommendations/stats
     * Get statistics about posts in the recommendation database
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        log.info("📊 Getting recommendation service statistics");
        
        try {
            Map<String, Object> stats = recommendationService.getStats();
            return ResponseEntity.ok(stats);
        } catch (Exception e) {
            log.error("❌ ERROR getting stats: {}", e.getMessage(), e);
            return ResponseEntity.ok(Map.of(
                "error", e.getMessage(),
                "timestamp", LocalDateTime.now()
            ));
        }
    }

    /**
     * POST /api/recommendations/sync-posts
     * Manually sync posts from post-service to recommendation database
     * Useful when Kafka events are missed
     */
    @PostMapping("/sync-posts")
    public ResponseEntity<Map<String, Object>> syncPosts(
            @RequestParam(defaultValue = "100") int limit) {
        log.info("🔄 Manual post sync requested, limit: {}", limit);
        
        try {
            int syncedCount = recommendationService.syncPostsFromPostService(limit);
            
            return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "Posts synced successfully",
                "syncedCount", syncedCount,
                "timestamp", LocalDateTime.now()
            ));
        } catch (Exception e) {
            log.error("❌ ERROR syncing posts: {}", e.getMessage(), e);
            return ResponseEntity.ok(Map.of(
                "success", false,
                "message", "Failed to sync posts: " + e.getMessage(),
                "timestamp", LocalDateTime.now()
            ));
        }
    }

    /**
     * GET /api/recommendations/user/{userId}/history
     * Get user's interaction history (debug endpoint)
     */
    @GetMapping("/user/{userId}/history")
    public ResponseEntity<Map<String, Object>> getUserHistory(
            @PathVariable String userId,
            @RequestParam(defaultValue = "30") int days) {
        log.info("📜 Getting interaction history for user: {}, days: {}", userId, days);
        
        try {
            Map<String, Object> history = recommendationService.getUserInteractionStats(userId, days);
            return ResponseEntity.ok(history);
        } catch (Exception e) {
            log.error("❌ ERROR getting user history: {}", e.getMessage(), e);
            return ResponseEntity.ok(Map.of(
                "error", e.getMessage(),
                "timestamp", LocalDateTime.now()
            ));
        }
    }

    /**
     * DELETE /api/recommendations/user/{userId}/history
     * Clear user's interaction history (allows re-recommendation of viewed posts)
     */
    @DeleteMapping("/user/{userId}/history")
    public ResponseEntity<Map<String, Object>> clearUserHistory(@PathVariable String userId) {
        log.info("🗑️ Clearing interaction history for user: {}", userId);
        
        try {
            int clearedCount = recommendationService.clearUserHistory(userId);
            
            // Also clear cache
            recommendationService.invalidateUserCache(userId);
            
            return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "User history cleared",
                "clearedInteractions", clearedCount,
                "timestamp", LocalDateTime.now()
            ));
        } catch (Exception e) {
            log.error("❌ ERROR clearing user history: {}", e.getMessage(), e);
            return ResponseEntity.ok(Map.of(
                "success", false,
                "message", "Failed to clear history: " + e.getMessage(),
                "timestamp", LocalDateTime.now()
            ));
        }
    }

    /**
     * GET /api/recommendations/health/python
     * Health check endpoint for Python model service
     */
    @GetMapping("/health/python")
    public ResponseEntity<Map<String, Object>> checkPythonServiceHealth() {
        // TODO: Implement actual health check via PythonModelServiceClient
        return ResponseEntity.ok(Map.of(
            "status", "checking",
            "message", "Python model service health check",
            "timestamp", LocalDateTime.now()
        ));
    }

    /**
     * DTO for interaction request
     */
    @Data
    public static class InteractionRequest {
        private String userId;
        private String postId;
        private String type;  // VIEW, LIKE, COMMENT, SHARE, SAVE, etc.
        private Double viewDuration;
        private Map<String, Object> context;
    }
}
