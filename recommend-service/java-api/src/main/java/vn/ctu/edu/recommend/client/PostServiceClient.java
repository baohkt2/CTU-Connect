package vn.ctu.edu.recommend.client;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import vn.ctu.edu.recommend.model.dto.PostDTO;

import java.util.List;

/**
 * Feign Client for Post Service
 * Communicates with post-service via Eureka service discovery
 */
@FeignClient(name = "post-service", fallback = PostServiceClientFallback.class)
public interface PostServiceClient {

    /**
     * Get post details by ID
     */
    @GetMapping("/api/posts/{id}")
    PostDTO getPostById(@PathVariable("id") String postId);

    /**
     * Get user's timeline posts
     */
    @GetMapping("/api/posts/timeline/{userId}")
    List<PostDTO> getUserTimeline(@PathVariable("userId") String userId, 
                                   @RequestParam(defaultValue = "0") int page,
                                   @RequestParam(defaultValue = "20") int size);

    /**
     * Get trending posts
     */
    @GetMapping("/api/posts/trending")
    List<PostDTO> getTrendingPosts(@RequestParam(defaultValue = "0") int page,
                                    @RequestParam(defaultValue = "20") int size);

    /**
     * Get posts by user ID
     */
    @GetMapping("/api/posts/user/{userId}")
    List<PostDTO> getPostsByUserId(@PathVariable("userId") String userId,
                                    @RequestParam(defaultValue = "0") int page,
                                    @RequestParam(defaultValue = "20") int size);

    /**
     * Get recent posts for feed
     */
    @GetMapping("/api/posts/feed")
    List<PostDTO> getFeedPosts(@RequestParam(defaultValue = "0") int page,
                               @RequestParam(defaultValue = "50") int size);
}
