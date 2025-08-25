package com.ctuconnect.controller;

import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.service.RecommendationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/recommendations")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = "*")
public class RecommendationController {

    private final RecommendationService recommendationService;

    /**
     * Lấy danh sách bài viết được gợi ý cá nhân hóa cho người dùng
     */
    @GetMapping("/personalized/{userId}")
    public ResponseEntity<List<PostResponse>> getPersonalizedRecommendations(
            @PathVariable String userId,
            @RequestParam(defaultValue = "10") int limit) {

        try {
            log.info("Getting personalized recommendations for user: {}, limit: {}", userId, limit);

            List<PostResponse> recommendations = recommendationService
                    .getPersonalizedRecommendations(userId, limit);

            log.info("Found {} personalized recommendations for user: {}",
                    recommendations.size(), userId);

            return ResponseEntity.ok(recommendations);

        } catch (Exception e) {
            log.error("Error getting personalized recommendations for user {}: {}", userId, e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Lấy danh sách bài viết tương tự dựa trên một bài viết
     */
    @GetMapping("/similar/{postId}")
    public ResponseEntity<List<PostResponse>> getSimilarPosts(
            @PathVariable String postId,
            @RequestParam(defaultValue = "5") int limit) {

        try {
            log.info("Getting similar posts for post: {}, limit: {}", postId, limit);

            List<PostResponse> similarPosts = recommendationService
                    .getSimilarPosts(postId, limit);

            log.info("Found {} similar posts for post: {}", similarPosts.size(), postId);

            return ResponseEntity.ok(similarPosts);

        } catch (Exception e) {
            log.error("Error getting similar posts for post {}: {}", postId, e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Lấy danh sách bài viết trending
     */
    @GetMapping("/trending")
    public ResponseEntity<List<PostResponse>> getTrendingPosts(
            @RequestParam(defaultValue = "10") int limit) {

        try {
            log.info("Getting trending posts, limit: {}", limit);

            List<PostResponse> trendingPosts = recommendationService.getTrendingPosts(limit);

            log.info("Found {} trending posts", trendingPosts.size());

            return ResponseEntity.ok(trendingPosts);

        } catch (Exception e) {
            log.error("Error getting trending posts: {}", e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Lấy gợi ý bài viết theo danh mục
     */
    @GetMapping("/category/{category}")
    public ResponseEntity<List<PostResponse>> getRecommendationsByCategory(
            @PathVariable String category,
            @RequestParam(required = false) String excludePostId,
            @RequestParam(defaultValue = "8") int limit) {

        try {
            log.info("Getting recommendations for category: {}, limit: {}, exclude: {}",
                    category, limit, excludePostId);

            List<PostResponse> recommendations = recommendationService
                    .getRecommendationsByCategory(category, excludePostId, limit);

            log.info("Found {} recommendations for category: {}", recommendations.size(), category);

            return ResponseEntity.ok(recommendations);

        } catch (Exception e) {
            log.error("Error getting recommendations for category {}: {}", category, e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Lấy gợi ý cho người dùng mới (chưa có lịch sử tương tác)
     */
    @GetMapping("/new-user/{userId}")
    public ResponseEntity<List<PostResponse>> getNewUserRecommendations(
            @PathVariable String userId,
            @RequestParam(defaultValue = "15") int limit) {

        try {
            log.info("Getting new user recommendations for: {}, limit: {}", userId, limit);

            List<PostResponse> recommendations = recommendationService
                    .getNewUserRecommendations(userId, limit);

            log.info("Found {} new user recommendations for: {}", recommendations.size(), userId);

            return ResponseEntity.ok(recommendations);

        } catch (Exception e) {
            log.error("Error getting new user recommendations for {}: {}", userId, e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Làm mới cache gợi ý cho người dùng
     */
    @DeleteMapping("/cache/{userId}")
    public ResponseEntity<String> invalidateRecommendationCache(
            @PathVariable String userId) {

        try {
            log.info("Invalidating recommendation cache for user: {}", userId);

            recommendationService.invalidateRecommendationCache(userId);

            return ResponseEntity.ok("Cache invalidated successfully for user: " + userId);

        } catch (Exception e) {
            log.error("Error invalidating cache for user {}: {}", userId, e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Endpoint tổng hợp cho trang chủ - kết hợp nhiều loại gợi ý
     */
    @GetMapping("/feed/{userId}")
    public ResponseEntity<RecommendationFeedResponse> getRecommendationFeed(
            @PathVariable String userId,
            @RequestParam(defaultValue = "20") int limit) {

        try {
            log.info("Getting recommendation feed for user: {}, limit: {}", userId, limit);

            // Lấy gợi ý cá nhân hóa (60% nội dung)
            int personalizedLimit = (int) (limit * 0.6);
            List<PostResponse> personalizedPosts = recommendationService
                    .getPersonalizedRecommendations(userId, personalizedLimit);

            // Lấy trending posts (40% nội dung)
            int trendingLimit = limit - personalizedPosts.size();
            List<PostResponse> trendingPosts = recommendationService
                    .getTrendingPosts(trendingLimit);

            // Nếu không đủ bài viết cá nhân hóa, lấy thêm từ new user recommendations
            if (personalizedPosts.size() < personalizedLimit) {
                int additionalLimit = personalizedLimit - personalizedPosts.size();
                List<PostResponse> additionalPosts = recommendationService
                        .getNewUserRecommendations(userId, additionalLimit);

                // Lọc bỏ bài viết trùng lặp
                Set<String> existingPostIds = personalizedPosts.stream()
                        .map(PostResponse::getId)
                        .collect(Collectors.toSet());

                additionalPosts.removeIf(post -> existingPostIds.contains(post.getId()));
                personalizedPosts.addAll(additionalPosts);
            }

            RecommendationFeedResponse response = RecommendationFeedResponse.builder()
                    .personalizedPosts(personalizedPosts)
                    .trendingPosts(trendingPosts)
                    .totalPosts(personalizedPosts.size() + trendingPosts.size())
                    .userId(userId)
                    .generatedAt(java.time.LocalDateTime.now())
                    .build();

            log.info("Generated recommendation feed for user: {} with {} total posts",
                    userId, response.getTotalPosts());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error getting recommendation feed for user {}: {}", userId, e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    // Response DTO cho feed tổng hợp
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class RecommendationFeedResponse {
        private String userId;
        private List<PostResponse> personalizedPosts;
        private List<PostResponse> trendingPosts;
        private int totalPosts;
        private java.time.LocalDateTime generatedAt;
    }
}
