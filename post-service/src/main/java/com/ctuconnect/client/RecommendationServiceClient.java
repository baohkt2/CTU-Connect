package com.ctuconnect.client;

import com.ctuconnect.config.FeignConfig;
import com.ctuconnect.dto.response.RecommendationFeedResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

/**
 * Feign Client for Recommendation Service
 * Communicates with recommendation-service to get AI-powered personalized feed
 */
@FeignClient(
    name = "recommendation-service",
    configuration = FeignConfig.class,
    fallback = RecommendationServiceClientFallback.class
)
public interface RecommendationServiceClient {

    /**
     * Get personalized feed recommendations for a user
     * @param userId User ID
     * @param page Page number (default 0)
     * @param size Number of recommendations to fetch (default 20)
     * @return RecommendationFeedResponse containing list of recommended posts with scores
     */
    @GetMapping("/api/recommendations/feed")
    RecommendationFeedResponse getRecommendationFeed(
        @RequestParam("userId") String userId,
        @RequestParam(value = "page", required = false, defaultValue = "0") Integer page,
        @RequestParam(value = "size", required = false, defaultValue = "20") Integer size
    );
}
