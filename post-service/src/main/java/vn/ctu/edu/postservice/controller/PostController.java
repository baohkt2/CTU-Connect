package vn.ctu.edu.postservice.controller;

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
import vn.ctu.edu.postservice.dto.request.CommentRequest;
import vn.ctu.edu.postservice.dto.request.InteractionRequest;
import vn.ctu.edu.postservice.dto.request.PostRequest;
import vn.ctu.edu.postservice.dto.response.CommentResponse;
import vn.ctu.edu.postservice.dto.response.InteractionResponse;
import vn.ctu.edu.postservice.dto.response.PostResponse;
import vn.ctu.edu.postservice.security.SecurityContextHolder;
import vn.ctu.edu.postservice.security.annotation.RequireAuth;
import vn.ctu.edu.postservice.service.CommentService;
import vn.ctu.edu.postservice.service.InteractionService;
import vn.ctu.edu.postservice.service.PostService;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/posts")
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:3001"}, allowCredentials = "true")
public class PostController {

    @Autowired
    private PostService postService;

    @Autowired
    private CommentService commentService;

    @Autowired
    private InteractionService interactionService;

    // Create post
    @PostMapping
    @RequireAuth
    public ResponseEntity<?> createPost(
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

    // Alternative endpoint for JSON-only post creation (without files)
    @PostMapping("/simple")
    @RequireAuth
    public ResponseEntity<?> createSimplePost(@Valid @RequestBody PostRequest request) {
        try {
            String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
            PostResponse response = postService.createPost(request, null, currentUserId);
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

    // Get all posts with pagination
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

    // Get post by ID (auto-record VIEW interaction)
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

    // Update post (author only)
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

    // Delete post (author only)
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

    // Add comment to post
    @PostMapping("/{id}/comments")
    public ResponseEntity<CommentResponse> addComment(
            @PathVariable String id,
            @Valid @RequestBody CommentRequest request) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        try {
            CommentResponse comment = commentService.createComment(id, request, currentUserId);
            return ResponseEntity.status(HttpStatus.CREATED).body(comment);
        } catch (RuntimeException e) {
            return ResponseEntity.badRequest().build();
        }
    }

    // Get comments for post
    @GetMapping("/{id}/comments")
    public ResponseEntity<Page<CommentResponse>> getComments(
            @PathVariable String id,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").ascending());
        Page<CommentResponse> comments = commentService.getCommentsByPost(id, pageable);
        return ResponseEntity.ok(comments);
    }

    // Record interaction (LIKE/SHARE/BOOKMARK)
    @PostMapping("/{id}/interactions")
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

    // Check if user has liked post
    @GetMapping("/{id}/likes/check")
    public ResponseEntity<Boolean> hasUserLikedPost(
            @PathVariable String id) {
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        boolean hasLiked = interactionService.hasUserReacted(id, userId);
        return ResponseEntity.ok(hasLiked);
    }

    // Get trending posts
    @GetMapping("/trending")
    public ResponseEntity<?> getTrendingPosts() {
        try {
            List<PostResponse> posts = postService.getTopViewedPosts();
            return ResponseEntity.ok(posts);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve trending posts", "message", e.getMessage()));
        }
    }

    // Get top viewed posts
    @GetMapping("/top-viewed")
    public ResponseEntity<?> getTopViewedPosts() {
        try {
            List<PostResponse> posts = postService.getTopViewedPosts();
            return ResponseEntity.ok(posts);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve top viewed posts", "message", e.getMessage()));
        }
    }

    // Get top liked posts
    @GetMapping("/top-liked")
    public ResponseEntity<?> getTopLikedPosts() {
        try {
            List<PostResponse> posts = postService.getTopLikedPosts();
            return ResponseEntity.ok(posts);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Failed to retrieve top liked posts", "message", e.getMessage()));
        }
    }

}
