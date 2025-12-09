package com.ctuconnect.client;

import com.ctuconnect.dto.response.RecommendationFeedResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.Collections;

/**
 * Fallback implementation for RecommendationServiceClient
 * Returns empty response when recommendation-service is unavailable
 */
@Slf4j
@Component
public class RecommendationServiceClientFallback implements RecommendationServiceClient {

    @Override
    public RecommendationFeedResponse getRecommendationFeed(String userId, Integer page, Integer size) {
        log.warn("⚠️  Recommendation service unavailable - using fallback for user: {}", userId);
        
        // Return empty response - post-service will fall back to regular posts
        return RecommendationFeedResponse.builder()
            .userId(userId)
            .recommendations(Collections.emptyList())
            .totalCount(0)
            .page(page)
            .size(size)
            .timestamp(LocalDateTime.now())
            .processingTimeMs(0L)
            .build();
    }
}
