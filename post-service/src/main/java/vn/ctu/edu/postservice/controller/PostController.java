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
import vn.ctu.edu.postservice.dto.request.CreateCommentRequest;
import vn.ctu.edu.postservice.dto.request.CreateInteractionRequest;
import vn.ctu.edu.postservice.dto.request.CreatePostRequest;
import vn.ctu.edu.postservice.dto.request.UpdatePostRequest;
import vn.ctu.edu.postservice.dto.response.CommentResponse;
import vn.ctu.edu.postservice.dto.response.InteractionResponse;
import vn.ctu.edu.postservice.dto.response.PostResponse;
import vn.ctu.edu.postservice.service.CommentService;
import vn.ctu.edu.postservice.service.InteractionService;
import vn.ctu.edu.postservice.service.PostService;

import java.util.List;

@RestController
@RequestMapping("/api/posts")
@CrossOrigin(origins = "*")
public class PostController {

    @Autowired
    private PostService postService;

    @Autowired
    private CommentService commentService;

    @Autowired
    private InteractionService interactionService;

    // Create post
    @PostMapping
    public ResponseEntity<PostResponse> createPost(
            @Valid @RequestPart("post") CreatePostRequest request,
            @RequestPart(value = "files", required = false) List<MultipartFile> files) {
        try {
            PostResponse response = postService.createPost(request, files);
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    // Get all posts with pagination
    @GetMapping
    public ResponseEntity<Page<PostResponse>> getAllPosts(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "desc") String sortDir,
            @RequestParam(required = false) String authorId,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String search) {

        Sort sort = sortDir.equalsIgnoreCase("desc") ?
            Sort.by(sortBy).descending() : Sort.by(sortBy).ascending();
        Pageable pageable = PageRequest.of(page, size, sort);

        Page<PostResponse> posts;

        if (search != null && !search.trim().isEmpty()) {
            posts = postService.searchPosts(search, pageable);
        } else if (authorId != null) {
            posts = postService.getPostsByAuthor(authorId, pageable);
        } else if (category != null) {
            posts = postService.getPostsByCategory(category, pageable);
        } else {
            posts = postService.getAllPosts(pageable);
        }

        return ResponseEntity.ok(posts);
    }

    // Get post by ID (auto-record VIEW interaction)
    @GetMapping("/{id}")
    public ResponseEntity<PostResponse> getPostById(
            @PathVariable String id,
            @RequestParam(required = false) String userId) {
        try {
            PostResponse post = postService.getPostById(id, userId);
            return ResponseEntity.ok(post);
        } catch (RuntimeException e) {
            return ResponseEntity.notFound().build();
        }
    }

    // Update post (author only)
    @PutMapping("/{id}")
    public ResponseEntity<PostResponse> updatePost(
            @PathVariable String id,
            @Valid @RequestBody UpdatePostRequest request,
            @RequestParam String authorId) {
        try {
            PostResponse updatedPost = postService.updatePost(id, request, authorId);
            return ResponseEntity.ok(updatedPost);
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }
    }

    // Delete post (author only)
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deletePost(
            @PathVariable String id,
            @RequestParam String authorId) {
        try {
            postService.deletePost(id, authorId);
            return ResponseEntity.noContent().build();
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }
    }

    // Add comment to post
    @PostMapping("/{id}/comments")
    public ResponseEntity<CommentResponse> addComment(
            @PathVariable String id,
            @Valid @RequestBody CreateCommentRequest request) {
        try {
            CommentResponse comment = commentService.createComment(id, request);
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
            @Valid @RequestBody CreateInteractionRequest request) {
        try {
            InteractionResponse interaction = interactionService.createInteraction(id, request);
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
            @PathVariable String id,
            @RequestParam String userId) {
        boolean hasLiked = interactionService.hasUserLikedPost(id, userId);
        return ResponseEntity.ok(hasLiked);
    }

    // Check if user has bookmarked post
    @GetMapping("/{id}/bookmarks/check")
    public ResponseEntity<Boolean> hasUserBookmarkedPost(
            @PathVariable String id,
            @RequestParam String userId) {
        boolean hasBookmarked = interactionService.hasUserBookmarkedPost(id, userId);
        return ResponseEntity.ok(hasBookmarked);
    }
}
