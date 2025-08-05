package com.ctuconnect.controller;

import com.ctuconnect.client.UserServiceClient;
import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.entity.InteractionEntity;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import com.ctuconnect.dto.request.CommentRequest;
import com.ctuconnect.dto.request.InteractionRequest;
import com.ctuconnect.dto.request.PostRequest;
import com.ctuconnect.dto.request.ScheduledPostRequest;
import com.ctuconnect.dto.response.CommentResponse;
import com.ctuconnect.dto.response.InteractionResponse;
import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.dto.response.PostAnalyticsResponse;
import com.ctuconnect.security.SecurityContextHolder;
import com.ctuconnect.security.annotation.RequireAuth;
import com.ctuconnect.security.AuthenticatedUser;
import com.ctuconnect.service.CommentService;
import com.ctuconnect.service.InteractionService;
import com.ctuconnect.service.PostService;
import com.ctuconnect.service.NewsFeedService;
import com.ctuconnect.service.NotificationService;
import com.ctuconnect.service.UserSyncService;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/posts")
public class PostController {

    @Autowired
    private PostService postService;

    @Autowired
    private CommentService commentService;

    @Autowired
    private InteractionService interactionService;

    @Autowired(required = false)
    private NewsFeedService newsFeedService;

    @Autowired(required = false)
    private NotificationService notificationService;

    @Autowired
    private UserSyncService userSyncService;
    @Autowired
    private UserServiceClient userServiceClient;

    // ========== ENHANCED ENDPOINTS (Primary) ==========

    /**
     * Create post - Primary endpoint (Enhanced)
     */
    @PostMapping
    @RequireAuth
    public ResponseEntity<?> createPost(@Valid @RequestBody PostRequest request) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            
            // Try enhanced service first, fallback to regular service
            PostResponse response;
            try {
                AuthenticatedUser user = new AuthenticatedUser(currentUserId, null, null);
                response = postService.createEnhancedPost(request, user);
            } catch (Exception e) {
                // Fallback to regular post creation
                response = postService.createPost(request, null, currentUserId);
            }
            
            // Invalidate caches if newsFeedService is available
            if (newsFeedService != null) {
                try {
                    newsFeedService.invalidateFeedCacheForUsers(
                        postService.getAffectedUserIds(response.getId()));
                } catch (Exception e) {
                    // Log error but don't fail the request
                    System.err.println("Failed to invalidate cache: " + e.getMessage());
                }
            }
            
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to create post", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Internal server error", "message", "Failed to create post"));
        }
    }

    /**
     * Facebook-like personalized news feed
     */
    @GetMapping("/feed")
    @RequireAuth
    public ResponseEntity<?> getPersonalizedFeed(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            
            if (newsFeedService != null) {
                List<PostResponse> feed = newsFeedService.generatePersonalizedFeed(
                    currentUserId, page, size);
                return ResponseEntity.ok(feed);
            } else {
                // Fallback to regular posts
                Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
                Page<PostResponse> posts = postService.getAllPosts(pageable);
                return ResponseEntity.ok(posts.getContent());
            }
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve feed", "message", e.getMessage()));
        }
    }

    /**
     * Get trending posts - Unified endpoint
     */
    @GetMapping("/trending")
    public ResponseEntity<?> getTrendingPosts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        
        try {
            List<PostResponse> posts;
            
            if (newsFeedService != null) {
                posts = newsFeedService.getTrendingPosts(page, size);
            } else {
                // Fallback to top viewed posts
                posts = postService.getTopViewedPosts();
            }
            
            return ResponseEntity.ok(posts);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve trending posts", "message", e.getMessage()));
        }
    }

    /**
     * Get user timeline (profile posts)
     */
    @GetMapping("/timeline/{userId}")
    @RequireAuth
    public ResponseEntity<?> getUserTimeline(
            @PathVariable String userId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            
            if (newsFeedService != null) {
                List<PostResponse> timeline = newsFeedService.generateUserTimeline(
                    userId, currentUserId, page, size);
                return ResponseEntity.ok(timeline);
            } else {
                // Fallback to posts by author
                Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
                Page<PostResponse> posts = postService.getPostsByAuthor(userId, pageable);
                return ResponseEntity.ok(posts.getContent());
            }
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve timeline", "message", e.getMessage()));
        }
    }

    // ========== LEGACY ENDPOINTS (Maintained for compatibility) ==========

    /**
     * Create post with file upload support
     */
    @PostMapping("/upload")
    @RequireAuth
    public ResponseEntity<?> createPostWithFiles(
            @Valid @RequestPart("post") PostRequest request,
            @RequestPart(value = "files", required = false) List<MultipartFile> files) {
        
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            PostResponse response = postService.createPost(request, files, currentUserId);
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to create post", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Internal server error", "message", "Failed to create post"));
        }
    }

    // ========== COMMON ENDPOINTS ==========

    /**
     * Get all posts with pagination and filters
     */
    @GetMapping
    public ResponseEntity<?> getAllPosts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "desc") String sortDir,
            @RequestParam(required = false) String authorId,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String search) {

        try {
            Sort sort = sortDir.equalsIgnoreCase("desc") ?
                Sort.by(sortBy).descending() : Sort.by(sortBy).ascending();
            Pageable pageable = PageRequest.of(page, size, sort);

            Page<PostResponse> posts;

            if (search != null && !search.trim().isEmpty()) {
                posts = postService.searchPosts(search.trim(), pageable);
            } else if (authorId != null && !authorId.trim().isEmpty()) {
                posts = postService.getPostsByAuthor(authorId.trim(), pageable);
            } else if (category != null && !category.trim().isEmpty()) {
                posts = postService.getPostsByCategory(category.trim(), pageable);
            } else {
                posts = postService.getAllPosts(pageable);
            }

            return ResponseEntity.ok(posts);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve posts", "message", e.getMessage()));
        }
    }

    @GetMapping("/me")
    @RequireAuth
    public ResponseEntity<?> getMyPosts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            System.out.println("DEBUG: Getting posts for current user ID: " + currentUserId);

            Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
            Page<PostResponse> posts = postService.getPostsByAuthor(currentUserId, pageable);

            System.out.println("DEBUG: Found " + posts.getTotalElements() + " posts for user " + currentUserId);

            // Additional debugging - log first few post author IDs
            posts.getContent().stream().limit(3).forEach(post -> {
                System.out.println("DEBUG: Post ID: " + post.getId() + ", Author ID: " +
                    (post.getAuthor() != null ? post.getAuthor().getId() : "null"));
            });

            return ResponseEntity.ok(posts);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            System.err.println("ERROR in getMyPosts: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve posts", "message", e.getMessage()));
        }
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<?> getUserPosts(
            @PathVariable String userId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size)
    {
        try {
            Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
            Page<PostResponse> posts = postService.getPostsByAuthor(userId, pageable);
            return ResponseEntity.ok(posts);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve user posts", "message", e.getMessage()));
        }
    }

    /**
     * Get post by ID (auto-record VIEW interaction)
     */
    @GetMapping("/{id}")
    public ResponseEntity<?> getPostById(@PathVariable String id) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserId();
            PostResponse post = postService.getPostById(id, currentUserId);
            return ResponseEntity.ok(post);
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", "Post not found", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve post", "message", e.getMessage()));
        }
    }

    /**
     * Update post (author only)
     */
    @PutMapping("/{id}")
    @RequireAuth
    public ResponseEntity<?> updatePost(
            @PathVariable String id,
            @Valid @RequestBody PostRequest request) {
        try {
            String authorId = SecurityContextHolder.getCurrentUserIdOrThrow();
            PostResponse updatedPost = postService.updatePost(id, request, authorId);
            return ResponseEntity.ok(updatedPost);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (RuntimeException e) {
            String message = e.getMessage();
            if (message.contains("Only the author")) {
                return ResponseEntity.status(HttpStatus.FORBIDDEN)
                        .body(Map.of("error", "Access denied", "message", message));
            }
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", "Post not found", "message", message));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to update post", "message", e.getMessage()));
        }
    }

    /**
     * Delete post (author only)
     */
    @DeleteMapping("/{id}")
    @RequireAuth
    public ResponseEntity<?> deletePost(@PathVariable String id) {
        try {
            String authorId = SecurityContextHolder.getCurrentUserIdOrThrow();
            postService.deletePost(id, authorId);
            return ResponseEntity.noContent().build();
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (RuntimeException e) {
            String message = e.getMessage();
            if (message.contains("Only the author")) {
                return ResponseEntity.status(HttpStatus.FORBIDDEN)
                        .body(Map.of("error", "Access denied", "message", message));
            }
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", "Post not found", "message", message));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to delete post", "message", e.getMessage()));
        }
    }

    // ========== INTERACTION ENDPOINTS ==========

    /**
     * Enhanced post interaction (like, comment, share)
     */
    @PostMapping("/{postId}/interact")
    @RequireAuth
    public ResponseEntity<?> interactWithPost(
            @PathVariable String postId,
            @RequestParam String action, // LIKE, UNLIKE, SHARE
            @RequestParam(required = false) String reactionType) {
        
        try {
            String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
            
            // Try enhanced interaction first
            try {
                postService.handlePostInteraction(postId, userId, action, reactionType);
                
                // Create notification for post author if services are available
                if (!"UNLIKE".equals(action) && notificationService != null) {
                    try {
                        String authorId = postService.getPostAuthorId(postId);
                        notificationService.createNotification(
                            authorId,
                            userId,
                            "POST_" + action,
                            "POST",
                            postId,
                            "User " + action.toLowerCase() + "d your post"
                        );
                    } catch (Exception e) {
                        // Log error but don't fail the request
                        System.err.println("Failed to create notification: " + e.getMessage());
                    }
                }
                
                return ResponseEntity.ok().build();
            } catch (Exception e) {
                // Fallback to legacy interaction handling
                InteractionRequest request = new InteractionRequest();
                // Set appropriate interaction type based on action
                request.setType(mapActionToInteractionType(action));
                
                InteractionResponse interaction = interactionService.createInteraction(postId, request, userId);
                if (interaction == null) {
                    return ResponseEntity.noContent().build();
                }
                return ResponseEntity.status(HttpStatus.CREATED).body(interaction);
            }
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to process interaction", "message", e.getMessage()));
        }
    }

    /**
     * Add comment to post - Legacy endpoint
     */
    @PostMapping("/{id}/comments")
    @RequireAuth
    public ResponseEntity<?> addComment(
            @PathVariable String id,
            @Valid @RequestBody CommentRequest request) {
        
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            CommentResponse comment = commentService.createComment(id, request, currentUserId);
            
            // Create notification if service is available
            if (notificationService != null) {
                try {
                    String authorId = postService.getPostAuthorId(id);
                    notificationService.createNotification(
                        authorId,
                        currentUserId,
                        "POST_COMMENTED",
                        "POST",
                        id,
                        "User commented on your post"
                    );
                } catch (Exception e) {
                    // Log error but don't fail the request
                    System.err.println("Failed to create notification: " + e.getMessage());
                }
            }
            
            return ResponseEntity.status(HttpStatus.CREATED).body(comment);
        } catch (RuntimeException e) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "Failed to add comment", "message", e.getMessage()));
        }
    }

    /**
     * Add comment with string content - Enhanced endpoint
     */
    @PostMapping("/{postId}/comments/simple")
    @RequireAuth
    public ResponseEntity<?> addSimpleComment(
            @PathVariable String postId,
            @RequestBody String content) {
        
        try {
            String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
            
            // Try enhanced comment service first
            try {
                postService.addComment(postId, userId, content);
                
                // Create notification
                if (notificationService != null) {
                    try {
                        String authorId = postService.getPostAuthorId(postId);
                        notificationService.createNotification(
                            authorId,
                            userId,
                            "POST_COMMENTED",
                            "POST",
                            postId,
                            "User commented on your post"
                        );
                    } catch (Exception e) {
                        System.err.println("Failed to create notification: " + e.getMessage());
                    }
                }
                
                return ResponseEntity.ok().build();
            } catch (Exception e) {
                // Fallback to legacy comment service
                CommentRequest request = new CommentRequest();
                request.setContent(content);
                CommentResponse comment = commentService.createComment(postId, request, userId);
                return ResponseEntity.status(HttpStatus.CREATED).body(comment);
            }
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to add comment", "message", e.getMessage()));
        }
    }

    /**
     * Get comments for post
     */
    @GetMapping("/{id}/comments")
    public ResponseEntity<Page<CommentResponse>> getComments(
            @PathVariable String id,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").ascending());
        Page<CommentResponse> comments = commentService.getCommentsByPost(id, pageable);
        return ResponseEntity.ok(comments);
    }

    /**
     * Record interaction (LIKE/SHARE/BOOKMARK) - Legacy endpoint
     */
    @PostMapping("/{id}/interactions")
    @RequireAuth
    public ResponseEntity<InteractionResponse> recordInteraction(
            @PathVariable String id,
            @Valid @RequestBody InteractionRequest request) {
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        try {
            InteractionResponse interaction = interactionService.createInteraction(id, request, userId);
            if (interaction == null) {
                // Interaction was removed (e.g., unlike)
                return ResponseEntity.noContent().build();
            }
            return ResponseEntity.status(HttpStatus.CREATED).body(interaction);
        } catch (RuntimeException e) {
            return ResponseEntity.badRequest().build();
        }
    }

    /**
     * Toggle like on a post
     */
    @PostMapping("/{postId}/like")
    @RequireAuth
    public ResponseEntity<?> toggleLike(@PathVariable String postId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

            InteractionRequest request = new InteractionRequest();
            request.setReaction(InteractionEntity.InteractionType.LIKE);
            request.setReactionType(InteractionEntity.ReactionType.LIKE);

            InteractionResponse response = interactionService.createInteraction(postId, request, currentUserId);

            // Create notification if it's a new like
            if (response.isActive() && notificationService != null) {
                try {
                    String authorId = postService.getPostAuthorId(postId);
                    if (!authorId.equals(currentUserId)) { // Don't notify self
                        notificationService.createNotification(
                            authorId,
                            currentUserId,
                            "POST_LIKED",
                            "POST",
                            postId,
                            "User liked your post"
                        );
                    }
                } catch (Exception e) {
                    System.err.println("Failed to create notification: " + e.getMessage());
                }
            }

            return ResponseEntity.ok(response);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to toggle like", "message", e.getMessage()));
        }
    }

    /**
     * Toggle bookmark on a post
     */
    @PostMapping("/{postId}/bookmark")
    @RequireAuth
    public ResponseEntity<?> toggleBookmark(@PathVariable String postId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

            InteractionRequest request = new InteractionRequest();
            request.setReaction(InteractionEntity.InteractionType.BOOKMARK);
            request.setReactionType(InteractionEntity.ReactionType.BOOKMARK);

            InteractionResponse response = interactionService.createInteraction(postId, request, currentUserId);
            return ResponseEntity.ok(response);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to toggle bookmark", "message", e.getMessage()));
        }
    }

    /**
     * Check if user has liked a post
     */
    @GetMapping("/{postId}/likes/check")
    @RequireAuth
    public ResponseEntity<?> checkUserLike(@PathVariable String postId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            boolean hasLiked = interactionService.hasUserLikedPost(postId, currentUserId);
            return ResponseEntity.ok(hasLiked);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to check like status", "message", e.getMessage()));
        }
    }

    /**
     * Check if user has bookmarked a post
     */
    @GetMapping("/{postId}/bookmarks/check")
    @RequireAuth
    public ResponseEntity<?> checkUserBookmark(@PathVariable String postId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            boolean hasBookmarked = interactionService.hasUserBookmarkedPost(postId, currentUserId);
            return ResponseEntity.ok(hasBookmarked);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to check bookmark status", "message", e.getMessage()));
        }
    }

    /**
     * Get user's interaction status for a post
     */
    @GetMapping("/{postId}/interactions/status")
    @RequireAuth
    public ResponseEntity<?> getUserInteractionStatus(@PathVariable String postId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            InteractionResponse response = interactionService.getUserInteractionStatus(postId, currentUserId);
            return ResponseEntity.ok(response);
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to get interaction status", "message", e.getMessage()));
        }
    }

    // ========== USER PROFILE SYNCHRONIZATION ENDPOINTS ==========

    /**
     * Manual user profile synchronization endpoint
     * This endpoint allows manual synchronization of user profile data across post-service
     */
    @PostMapping("/admin/sync-user-profile/{userId}")
    @RequireAuth(roles = {"ADMIN"})
    public ResponseEntity<?> syncUserProfile(@PathVariable String userId) {
        try {
            // Get updated user info from user-service
            AuthorInfo updatedAuthor = userServiceClient.getAuthorInfo(userId);
            if (updatedAuthor == null) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND)
                        .body(Map.of("error", "User not found", "userId", userId));
            }

            // Perform manual sync
            userSyncService.manualSyncUserProfile(userId, updatedAuthor);

            return ResponseEntity.ok(Map.of(
                "message", "User profile synchronized successfully",
                "userId", userId,
                "updatedAuthor", updatedAuthor
            ));
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Admin access required"));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to sync user profile", "message", e.getMessage()));
        }
    }

    // ========== HELPER METHODS ==========

    private String mapActionToInteractionType(String action) {
        switch (action.toUpperCase()) {
            case "LIKE":
                return "LIKE";
            case "SHARE":
                return "SHARE";
            case "BOOKMARK":
                return "BOOKMARK";
            default:
                return "LIKE";
        }
    }
}
