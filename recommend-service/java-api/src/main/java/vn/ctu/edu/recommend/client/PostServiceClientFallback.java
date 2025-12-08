package vn.ctu.edu.recommend.client;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.model.dto.PostDTO;

import java.util.Collections;
import java.util.List;

/**
 * Fallback implementation for Post Service Client
 * Returns empty/default responses when post-service is unavailable
 */
@Component
@Slf4j
public class PostServiceClientFallback implements PostServiceClient {

    @Override
    public PostDTO getPostById(String postId) {
        log.warn("Post Service unavailable. Returning null for post: {}", postId);
        return null;
    }

    @Override
    public List<PostDTO> getUserTimeline(String userId, int page, int size) {
        log.warn("Post Service unavailable. Returning empty timeline for user: {}", userId);
        return Collections.emptyList();
    }

    @Override
    public List<PostDTO> getTrendingPosts(int page, int size) {
        log.warn("Post Service unavailable. Returning empty trending posts");
        return Collections.emptyList();
    }

    @Override
    public List<PostDTO> getPostsByUserId(String userId, int page, int size) {
        log.warn("Post Service unavailable. Returning empty posts for user: {}", userId);
        return Collections.emptyList();
    }

    @Override
    public List<PostDTO> getFeedPosts(int page, int size) {
        log.warn("Post Service unavailable. Returning empty feed posts");
        return Collections.emptyList();
    }
}
