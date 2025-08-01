package com.ctuconnect.controller;

import com.ctuconnect.dto.request.ScheduledPostRequest;
import com.ctuconnect.dto.response.PostAnalyticsResponse;
import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.dto.request.PostRequest;
import com.ctuconnect.service.NewsFeedService;
import com.ctuconnect.service.PostService;
import com.ctuconnect.service.NotificationService;
import com.ctuconnect.security.annotation.RequireAuth;
import com.ctuconnect.security.AuthenticatedUser;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/enhanced-posts")
@RequiredArgsConstructor
public class EnhancedPostController {

    private final PostService postService;
    private final NewsFeedService newsFeedService;
    private final NotificationService notificationService;

    /**
     * Facebook-like personalized news feed
     */
    @GetMapping("/feed")
    @RequireAuth
    public ResponseEntity<List<PostResponse>> getPersonalizedFeed(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            AuthenticatedUser user) {

        List<PostResponse> feed = newsFeedService.generatePersonalizedFeed(
            user.getId(), page, size);

        return ResponseEntity.ok(feed);
    }

    /**
     * Get trending posts
     */
    @GetMapping("/trending")
    public ResponseEntity<List<PostResponse>> getTrendingPosts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        List<PostResponse> trendingPosts = newsFeedService.getTrendingPosts(page, size);
        return ResponseEntity.ok(trendingPosts);
    }

    /**
     * Get user timeline (profile posts)
     */
    @GetMapping("/timeline/{userId}")
    @RequireAuth
    public ResponseEntity<List<PostResponse>> getUserTimeline(
            @PathVariable String userId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            AuthenticatedUser viewer) {

        List<PostResponse> timeline = newsFeedService.generateUserTimeline(
            userId, viewer.getId(), page, size);

        return ResponseEntity.ok(timeline);
    }

    /**
     * Create post with enhanced features
     */
    @PostMapping
    @RequireAuth
    public ResponseEntity<PostResponse> createPost(
            @RequestBody PostRequest request,
            AuthenticatedUser user) {

        PostResponse post = postService.createEnhancedPost(request, user);

        // Invalidate relevant caches
        newsFeedService.invalidateFeedCacheForUsers(
            postService.getAffectedUserIds(post.getId()));

        return ResponseEntity.ok(post);
    }

    /**
     * Enhanced post interaction (like, comment, share)
     */
    @PostMapping("/{postId}/interact")
    @RequireAuth
    public ResponseEntity<Void> interactWithPost(
            @PathVariable String postId,
            @RequestParam String action, // LIKE, UNLIKE, SHARE
            @RequestParam(required = false) String reactionType,
            AuthenticatedUser user) {

        postService.handlePostInteraction(postId, user.getId(), action, reactionType);

        // Create notification for post author
        if (!"UNLIKE".equals(action)) {
            notificationService.createNotification(
                postService.getPostAuthorId(postId),
                user.getId(),
                "POST_" + action,
                "POST",
                postId,
                user.getFullName() + " " + action.toLowerCase() + "d your post"
            );
        }

        return ResponseEntity.ok().build();
    }

    /**
     * Add comment to post
     */
    @PostMapping("/{postId}/comments")
    @RequireAuth
    public ResponseEntity<Void> addComment(
            @PathVariable String postId,
            @RequestBody String content,
            AuthenticatedUser user) {

        postService.addComment(postId, user.getId(), content);

        // Create notification
        notificationService.createNotification(
            postService.getPostAuthorId(postId),
            user.getId(),
            "POST_COMMENTED",
            "POST",
            postId,
            user.getFullName() + " commented on your post"
        );

        return ResponseEntity.ok().build();
    }

    /**
     * Get post analytics (for post author)
     */
    @GetMapping("/{postId}/analytics")
    @RequireAuth
    public ResponseEntity<PostAnalyticsResponse> getPostAnalytics(
            @PathVariable String postId,
            AuthenticatedUser user) {

        PostAnalyticsResponse analytics = postService.getPostAnalytics(postId, user.getId());
        return ResponseEntity.ok(analytics);
    }

    /**
     * Schedule post for later publishing
     */
    @PostMapping("/schedule")
    @RequireAuth
    public ResponseEntity<PostResponse> schedulePost(
            @RequestBody ScheduledPostRequest request,
            AuthenticatedUser user) {

        PostResponse scheduledPost = postService.schedulePost(request, user);
        return ResponseEntity.ok(scheduledPost);
    }

    /**
     * Search posts with advanced filters
     */
    @GetMapping("/search")
    public ResponseEntity<List<PostResponse>> searchPosts(
            @RequestParam String query,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String faculty,
            @RequestParam(required = false) String dateRange,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        List<PostResponse> results = postService.searchPosts(
            query, category, faculty, dateRange, PageRequest.of(page, size));

        return ResponseEntity.ok(results);
    }
}
