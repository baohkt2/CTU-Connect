package vn.ctu.edu.recommend.service;

import vn.ctu.edu.recommend.model.dto.RecommendationRequest;
import vn.ctu.edu.recommend.model.dto.RecommendationResponse;
import vn.ctu.edu.recommend.model.dto.FeedbackRequest;

/**
 * Main recommendation service interface
 */
public interface RecommendationService {
    
    /**
     * Get personalized recommendations for a user
     */
    RecommendationResponse getRecommendations(RecommendationRequest request);
    
    /**
     * Record user feedback for reinforcement learning
     */
    void recordFeedback(FeedbackRequest request);
    
    /**
     * Rebuild embeddings for all posts
     */
    void rebuildEmbeddings();
    
    /**
     * Rebuild recommendation cache for all users
     */
    void rebuildRecommendationCache();
    
    /**
     * Invalidate cache for specific user
     */
    void invalidateUserCache(String userId);
}
