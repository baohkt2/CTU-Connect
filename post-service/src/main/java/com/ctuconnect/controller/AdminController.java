package com.ctuconnect.controller;

import com.ctuconnect.entity.InteractionEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.ctuconnect.dto.response.CommentResponse;
import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.repository.CommentRepository;
import com.ctuconnect.repository.InteractionRepository;
import com.ctuconnect.repository.PostRepository;
import com.ctuconnect.service.CommentService;
import com.ctuconnect.service.PostService;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin")
@CrossOrigin(origins = "*")
public class AdminController {

    @Autowired
    private PostService postService;

    @Autowired
    private CommentService commentService;

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private CommentRepository commentRepository;

    @Autowired
    private InteractionRepository interactionRepository;

    // Get all posts (admin view)
    @GetMapping("/posts")
    public ResponseEntity<Page<PostResponse>> getAllPostsAdmin(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(defaultValue = "createdAt") String sortBy,
            @RequestParam(defaultValue = "desc") String sortDir) {

        Sort sort = sortDir.equalsIgnoreCase("desc") ?
            Sort.by(sortBy).descending() : Sort.by(sortBy).ascending();
        Pageable pageable = PageRequest.of(page, size, sort);

        Page<PostResponse> posts = postService.getAllPosts(pageable);
        return ResponseEntity.ok(posts);
    }

    // Delete any post (admin)
    @DeleteMapping("/posts/{id}")
    public ResponseEntity<Void> deletePostAdmin(@PathVariable String id) {
        try {
            postRepository.deleteById(id);
            // Clean up associated data
            commentRepository.deleteByPostId(id);
            interactionRepository.deleteByPostId(id);
            return ResponseEntity.noContent().build();
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }

    // Get all comments (admin view)
    @GetMapping("/comments")
    public ResponseEntity<Page<CommentResponse>> getAllCommentsAdmin(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {

        Pageable pageable = PageRequest.of(page, size, Sort.by("createdAt").descending());
        Page<CommentResponse> comments = commentRepository.findAll(pageable)
                .map(CommentResponse::new);
        return ResponseEntity.ok(comments);
    }

    // Delete any comment (admin)
    @DeleteMapping("/comments/{id}")
    public ResponseEntity<Void> deleteCommentAdmin(@PathVariable String id) {
        try {
            commentRepository.deleteById(id);
            return ResponseEntity.noContent().build();
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }

    // Get metrics - top posts by views
    @GetMapping("/metrics/top-viewed-posts")
    public ResponseEntity<List<PostResponse>> getTopViewedPosts() {
        List<PostResponse> topPosts = postService.getTopViewedPosts();
        return ResponseEntity.ok(topPosts);
    }

    // Get metrics - top posts by likes
    @GetMapping("/metrics/top-liked-posts")
    public ResponseEntity<List<PostResponse>> getTopLikedPosts() {
        List<PostResponse> topPosts = postService.getTopLikedPosts();
        return ResponseEntity.ok(topPosts);
    }

    // Get user statistics
    @GetMapping("/metrics/user-stats/{userId}")
    public ResponseEntity<Map<String, Long>> getUserStats(@PathVariable String userId) {
        Map<String, Long> stats = new HashMap<>();
        stats.put("totalPosts", postRepository.countByAuthorId(userId));
        stats.put("totalComments", commentRepository.countByAuthorId(userId));
        stats.put("totalLikes", interactionRepository.countByUserIdAndType(userId,
                InteractionEntity.InteractionType.LIKE));
        stats.put("totalBookmarks", interactionRepository.countByUserIdAndType(userId,
                InteractionEntity.InteractionType.BOOKMARK));

        return ResponseEntity.ok(stats);
    }

    // Get overall platform statistics
    @GetMapping("/metrics/platform-stats")
    public ResponseEntity<Map<String, Long>> getPlatformStats() {
        Map<String, Long> stats = new HashMap<>();
        stats.put("totalPosts", postRepository.count());
        stats.put("totalComments", commentRepository.count());
        stats.put("totalInteractions", interactionRepository.count());

        return ResponseEntity.ok(stats);
    }
}
