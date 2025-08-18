package com.ctuconnect.controller;

import com.ctuconnect.dto.request.CommentRequest;
import com.ctuconnect.dto.response.CommentResponse;
import com.ctuconnect.security.SecurityContextHolder;
import com.ctuconnect.service.CommentService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/comments")
public class CommentController {

    @Autowired
    private CommentService commentService;

    /**
     * Create a new comment or reply with depth management
     */
    @PostMapping
    public ResponseEntity<?> createComment(
            @Valid @RequestBody CommentRequest request,
            @RequestParam String postId) {
        try {
            String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
            CommentResponse comment = commentService.createComment(postId, request, userId);

            // Send notification for new comment/reply
            try {
                if (request.getParentCommentId() != null) {
                    // This is a reply - notify parent comment author
                    CommentResponse parentComment = commentService.getCommentById(request.getParentCommentId());
                    if (!parentComment.getAuthor().getId().equals(userId)) {
                        // TODO: Implement notification service integration
                        System.out.println("Would send notification to: " + parentComment.getAuthor().getId());
                    }
                }
                // Root comments are handled in PostController for consistency
            } catch (Exception e) {
                // Log notification error but don't fail the comment creation
                System.err.println("Failed to send comment notification: " + e.getMessage());
            }

            return ResponseEntity.status(HttpStatus.CREATED).body(comment);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to create comment", "message", e.getMessage()));
        }
    }

    /**
     * Get all comments for a post with hierarchical structure
     */
    @GetMapping("/post/{postId}")
    public ResponseEntity<Page<CommentResponse>> getCommentsByPost(
            @PathVariable String postId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "asc") String sortDir) {
        try {
            Sort sort = sortDir.equalsIgnoreCase("desc") ?
                Sort.by(sortBy).descending() : Sort.by(sortBy).ascending();
            Pageable pageable = PageRequest.of(page, size, sort);

            Page<CommentResponse> comments = commentService.getCommentsByPost(postId, pageable);
            return ResponseEntity.ok(comments);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    /**
     * Get all replies for a specific comment (both nested and flattened)
     */
    @GetMapping("/{commentId}/replies")
    public ResponseEntity<List<CommentResponse>> getReplies(@PathVariable String commentId) {
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
    @GetMapping("/{id}")
    public ResponseEntity<CommentResponse> getCommentById(@PathVariable String id) {
        try {
            CommentResponse comment = commentService.getCommentById(id);
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
    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteComment(
            @PathVariable String id) {
        try {
            String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
            commentService.deleteComment(id, userId);
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
    @GetMapping("/post/{postId}/count")
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
     * Create a simple reply with just content (convenience endpoint)
     */
    @PostMapping("/{parentId}/reply")
    public ResponseEntity<?> createReply(
            @PathVariable String parentId,
            @RequestParam String postId,
            @RequestParam String content) {
        try {
            String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
            CommentRequest request = new CommentRequest();
            request.setContent(content);
            request.setParentCommentId(parentId);

            CommentResponse reply = commentService.createComment(postId, request, userId);

            // Send notification to parent comment author
            try {
                CommentResponse parentComment = commentService.getCommentById(parentId);
                if (!parentComment.getAuthor().getId().equals(userId)) {
                    // TODO: Implement notification service integration
                    System.out.println("Would send reply notification to: " + parentComment.getAuthor().getId());
                }
            } catch (Exception e) {
                System.err.println("Failed to send reply notification: " + e.getMessage());
            }

            return ResponseEntity.status(HttpStatus.CREATED).body(reply);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "Failed to create reply", "message", e.getMessage()));
        }
    }
}
