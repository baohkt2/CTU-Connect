package com.ctuconnect.controller;

import com.ctuconnect.client.UserServiceClient;
import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.dto.request.SearchRequest;
import com.ctuconnect.dto.response.SearchResponse;
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
import com.ctuconnect.service.SearchService;
import com.ctuconnect.service.PostInteractionService;

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

    @Autowired
    private SearchService searchService;

    @Autowired
    private PostInteractionService postInteractionService;

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

                AuthenticatedUser user = new AuthenticatedUser(currentUserId, null, null);
                response = postService.createEnhancedPost(request, user);

            
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
    /*@PostMapping("/upload")
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
*/
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
     * Add comment to post - Enhanced with depth management
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
                    if (request.getParentCommentId() != null) {
                        // This is a reply - notify parent comment author
                        CommentResponse parentComment = commentService.getCommentById(request.getParentCommentId());
                        if (!parentComment.getAuthor().getId().equals(currentUserId)) {
                            notificationService.createNotification(
                                parentComment.getAuthor().getId(),
                                currentUserId,
                                "COMMENT_REPLIED",
                                "COMMENT",
                                request.getParentCommentId(),
                                "Someone replied to your comment"
                            );
                        }
                    } else {
                        // This is a root comment - notify post author
                        String authorId = postService.getPostAuthorId(id);
                        if (!authorId.equals(currentUserId)) {
                            notificationService.createNotification(
                                authorId,
                                currentUserId,
                                "POST_COMMENTED",
                                "POST",
                                id,
                                "User commented on your post"
                            );
                        }
                    }
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
     * Get comments for post with hierarchical structure (Enhanced)
     */
    @GetMapping("/{id}/comments")
    public ResponseEntity<Page<CommentResponse>> getComments(
            @PathVariable String id,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "asc") String sortDir) {

        Sort sort = sortDir.equalsIgnoreCase("desc") ?
            Sort.by(sortBy).descending() : Sort.by(sortBy).ascending();
        Pageable pageable = PageRequest.of(page, size, sort);
        Page<CommentResponse> comments = commentService.getCommentsByPost(id, pageable);
        return ResponseEntity.ok(comments);
    }

    /**
     * Get all replies for a specific comment (both nested and flattened)
     */
    @GetMapping("/{postId}/comments/{commentId}/replies")
    public ResponseEntity<List<CommentResponse>> getCommentReplies(
            @PathVariable String postId,
            @PathVariable String commentId) {
        try {
            List<CommentResponse> replies = commentService.getReplies(commentId);
            return ResponseEntity.ok(replies);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(null);
        }
    }

    /**
     * Get a specific comment by ID
     */
    @GetMapping("/{postId}/comments/{commentId}")
    public ResponseEntity<CommentResponse> getCommentById(
            @PathVariable String postId,
            @PathVariable String commentId) {
        try {
            CommentResponse comment = commentService.getCommentById(commentId);
            return ResponseEntity.ok(comment);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Delete a comment and all its replies
     */
    @DeleteMapping("/{postId}/comments/{commentId}")
    @RequireAuth
    public ResponseEntity<?> deleteComment(
            @PathVariable String postId,
            @PathVariable String commentId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            commentService.deleteComment(commentId, currentUserId);
            return ResponseEntity.ok(Map.of("message", "Comment deleted successfully"));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of("error", "Unauthorized", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to delete comment", "message", e.getMessage()));
        }
    }

    /**
     * Get comment count for a post
     */
    @GetMapping("/{postId}/comments/count")
    public ResponseEntity<Map<String, Long>> getCommentCount(@PathVariable String postId) {
        try {
            long count = commentService.getCommentCountByPost(postId);
            return ResponseEntity.ok(Map.of("count", count));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("count", 0L));
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
            // Don't set reactionType for bookmark - it's not a reaction

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

    @PostMapping("/{postId}/share")
    @RequireAuth
    public ResponseEntity<?> toggleShare(@PathVariable String postId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

            InteractionRequest request = new InteractionRequest();
            request.setReaction(InteractionEntity.InteractionType.SHARE);
            // Don't set reactionType for bookmark - it's not a reaction

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

    // ========== COMMENT ENDPOINTS (Proxy to CommentService, used by client-frontend) ==========

    /**
     * Toggle like for a comment (stub/no-op for now)
     * Frontend performs optimistic update; backend can be enhanced later.
     */
    @PostMapping("/{postId}/comments/{commentId}/like")
    @RequireAuth
    public ResponseEntity<?> toggleCommentLike(
            @PathVariable String postId,
            @PathVariable String commentId) {
        try {
            String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
            return ResponseEntity.ok(Map.of(
                    "message", "Comment like toggled",
                    "postId", postId,
                    "commentId", commentId,
                    "userId", userId
            ));
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to toggle like", "message", e.getMessage()));
        }
    }

    // ========== ENHANCED SEARCH ENDPOINTS ==========

    /**
     * Advanced search endpoint with multiple filters
     */
    @PostMapping("/search/advanced")
    public ResponseEntity<?> advancedSearch(@RequestBody SearchRequest searchRequest) {
        try {
            long startTime = System.currentTimeMillis();
            SearchResponse response = searchService.searchPosts(searchRequest);
            long endTime = System.currentTimeMillis();

            response.setSearchTimeMs(endTime - startTime);
            response.setPageSize(searchRequest.getSize());

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Advanced search failed", "message", e.getMessage()));
        }
    }

    /**
     * Get search suggestions for auto-complete
     */
    @GetMapping("/search/suggestions")
    public ResponseEntity<?> getSearchSuggestions(
            @RequestParam String query) {

        try {
            if (query == null || query.trim().length() < 2) {
                return ResponseEntity.badRequest()
                    .body(Map.of("error", "Query must be at least 2 characters long"));
            }

            var suggestions = searchService.getSearchSuggestions(query.trim());
            return ResponseEntity.ok(suggestions);
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Failed to get suggestions", "message", e.getMessage()));
        }
    }

    /**
     * Get related posts for a specific post
     */
    @GetMapping("/{postId}/related")
    public ResponseEntity<?> getRelatedPosts(
            @PathVariable String postId,
            @RequestParam(defaultValue = "5") int limit) {

        try {
            List<PostResponse> relatedPosts = searchService.getRelatedPosts(postId, limit);
            return ResponseEntity.ok(Map.of("relatedPosts", relatedPosts));
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Failed to get related posts", "message", e.getMessage()));
        }
    }

    /**
     * Get trending search terms
     */
    @GetMapping("/search/trending")
    public ResponseEntity<?> getTrendingSearchTerms() {
        try {
            List<String> trendingTerms = searchService.getTrendingSearchTerms();
            return ResponseEntity.ok(Map.of("trendingTerms", trendingTerms));
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Failed to get trending terms", "message", e.getMessage()));
        }
    }

    /**
     * Record post view (for recommendation system)
     */
    @PostMapping("/{postId}/view")
    public ResponseEntity<?> recordPostView(
            @PathVariable String postId,
            @RequestParam(required = false) String userId,
            @RequestParam(required = false, defaultValue = "0") long readingTimeSeconds) {

        try {
            // Get user ID from authentication or parameter
            String viewerId = SecurityContextHolder.getCurrentUserId();
            if (viewerId == null && userId != null) {
                viewerId = userId;
            }

            if (viewerId != null) {
                // Record view interaction
                postInteractionService.recordPostView(viewerId, postId);

                // Record reading time if provided
                if (readingTimeSeconds > 0) {
                    postInteractionService.recordReadingTime(viewerId, postId, readingTimeSeconds);
                }
            }

            return ResponseEntity.ok(Map.of("message", "View recorded successfully"));

        } catch (Exception e) {
            // Don't fail the request if tracking fails
            return ResponseEntity.ok(Map.of("message", "View tracking failed but request processed"));
        }
    }

    /**
     * Enhanced like endpoint with interaction tracking
     */
    @PostMapping("/{postId}/like/enhanced")
    @RequireAuth
    public ResponseEntity<?> enhancedToggleLike(@PathVariable String postId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

            // Check current like status
            boolean currentlyLiked = interactionService.hasUserLikedPost(postId, currentUserId);

            // Toggle like using existing service
            InteractionRequest request = new InteractionRequest();
            request.setReaction(InteractionEntity.InteractionType.LIKE);
            request.setReactionType(InteractionEntity.ReactionType.LIKE);

            InteractionResponse response = interactionService.createInteraction(postId, request, currentUserId);

            // Record interaction for recommendation system
            boolean isLike = !currentlyLiked; // Toggled state
            postInteractionService.recordPostLike(currentUserId, postId, isLike);

            // Create notification if it's a new like
            if (isLike && notificationService != null) {
                try {
                    String authorId = postService.getPostAuthorId(postId);
                    if (!authorId.equals(currentUserId)) {
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
     * Enhanced comment endpoint with interaction tracking
     */
    @PostMapping("/{id}/comments/enhanced")
    @RequireAuth
    public ResponseEntity<?> addEnhancedComment(
            @PathVariable String id,
            @Valid @RequestBody CommentRequest request) {

        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

            // Add comment using existing service
            CommentResponse comment = commentService.createComment(id, request, currentUserId);

            // Record interaction for recommendation system
            postInteractionService.recordPostComment(currentUserId, id);

            // Create notification if service is available
            if (notificationService != null) {
                try {
                    if (request.getParentCommentId() != null) {
                        // Reply notification
                        CommentResponse parentComment = commentService.getCommentById(request.getParentCommentId());
                        if (!parentComment.getAuthor().getId().equals(currentUserId)) {
                            notificationService.createNotification(
                                parentComment.getAuthor().getId(),
                                currentUserId,
                                "COMMENT_REPLIED",
                                "COMMENT",
                                request.getParentCommentId(),
                                "Someone replied to your comment"
                            );
                        }
                    } else {
                        // Comment notification
                        String authorId = postService.getPostAuthorId(id);
                        if (!authorId.equals(currentUserId)) {
                            notificationService.createNotification(
                                authorId,
                                currentUserId,
                                "POST_COMMENTED",
                                "POST",
                                id,
                                "User commented on your post"
                            );
                        }
                    }
                } catch (Exception e) {
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
     * Enhanced share endpoint with interaction tracking
     */
    @PostMapping("/{postId}/share/enhanced")
    @RequireAuth
    public ResponseEntity<?> enhancedToggleShare(@PathVariable String postId) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

            // Use existing share logic
            InteractionRequest request = new InteractionRequest();
            request.setReaction(InteractionEntity.InteractionType.SHARE);

            InteractionResponse response = interactionService.createInteraction(postId, request, currentUserId);

            // Record interaction for recommendation system
            postInteractionService.recordPostShare(currentUserId, postId);

            return ResponseEntity.ok(response);

        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to toggle share", "message", e.getMessage()));
        }
    }

    /**
     * Get user's interaction preferences and history
     */
    @GetMapping("/interactions/preferences")
    @RequireAuth
    public ResponseEntity<?> getUserInteractionPreferences() {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

            // Get user's content preferences
            PostInteractionService.UserContentPreferences preferences =
                postInteractionService.getUserContentPreferences(currentUserId);

            // Get recent interaction history (last 30 days)
            Map<String, Double> interactionHistory =
                postInteractionService.getUserInteractionHistory(currentUserId, 30);

            return ResponseEntity.ok(Map.of(
                "preferences", preferences,
                "recentInteractions", interactionHistory.size(),
                "topCategories", preferences.getCategoryScores(),
                "topTags", preferences.getTagScores(),
                "topAuthors", preferences.getAuthorScores()
            ));

        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Authentication required", "message", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to get preferences", "message", e.getMessage()));
        }
    }
}
