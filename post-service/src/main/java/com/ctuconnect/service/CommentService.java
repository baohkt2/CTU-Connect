package com.ctuconnect.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import com.ctuconnect.client.UserServiceClient;
import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.dto.request.CommentRequest;
import com.ctuconnect.dto.response.CommentResponse;
import com.ctuconnect.entity.CommentEntity;
import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.repository.CommentRepository;
import com.ctuconnect.repository.PostRepository;

import java.util.*;
import java.util.stream.Collectors;


@Service
public class CommentService {

    @Autowired
    private CommentRepository commentRepository;

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private EventService eventService;

    @Autowired
    private UserServiceClient userServiceClient;

    /**
     * Create a new comment or reply with depth management and flattening strategy
     */
    public CommentResponse createComment(String postId, CommentRequest request, String authorId) {
        // Verify post exists
        AuthorInfo author = userServiceClient.getAuthorInfo(authorId);
        Optional<PostEntity> postOpt = postRepository.findById(postId);
        if (postOpt.isEmpty()) {
            throw new RuntimeException("Post not found with id: " + postId);
        }

        CommentEntity comment;

        if (request.getParentCommentId() != null) {
            // This is a reply - implement depth management
            comment = createReplyWithDepthManagement(postId, request, author);
        } else {
            // This is a root comment
            comment = new CommentEntity(postId, request.getContent(), author);
            comment.setDepth(0);
        }

        CommentEntity savedComment = commentRepository.save(comment);

        // Update post comment count
        PostEntity post = postOpt.get();
        post.getStats().incrementComments();
        postRepository.save(post);

        // Publish event
        eventService.publishCommentEvent("COMMENT_CREATED", postId, savedComment.getId(), savedComment.getAuthor().getId());

        return new CommentResponse(savedComment);
    }

    /**
     * Create a reply with proper depth management and flattening
     */
    private CommentEntity createReplyWithDepthManagement(String postId, CommentRequest request, AuthorInfo author) {
        Optional<CommentEntity> parentOpt = commentRepository.findById(request.getParentCommentId());
        if (parentOpt.isEmpty()) {
            throw new RuntimeException("Parent comment not found with id: " + request.getParentCommentId());
        }

        CommentEntity parent = parentOpt.get();
        int parentDepth = parent.getDepth() != null ? parent.getDepth() : 0;
        int newDepth = parentDepth + 1;

        CommentEntity reply;

        if (newDepth >= CommentEntity.MAX_DEPTH) {
            // Flatten: attach to root comment instead of parent
            String rootCommentId = parent.getRootCommentId() != null ?
                parent.getRootCommentId() :
                (parent.isRootComment() ? parent.getId() : request.getParentCommentId());

            String replyToAuthor = parent.getAuthor().getFullName() != null ?
                parent.getAuthor().getFullName() : parent.getAuthor().getName();

            reply = new CommentEntity(
                postId,
                request.getContent(),
                author,
                rootCommentId, // parentCommentId becomes rootCommentId
                rootCommentId, // rootCommentId
                CommentEntity.MAX_DEPTH, // depth
                replyToAuthor // replyToAuthor
            );
        } else {
            // Normal nested reply
            String rootCommentId = parent.getRootCommentId() != null ?
                parent.getRootCommentId() :
                (parent.isRootComment() ? parent.getId() : null);

            reply = new CommentEntity(
                postId,
                request.getContent(),
                author,
                request.getParentCommentId(),
                rootCommentId,
                newDepth,
                null // no replyToAuthor for normal nested replies
            );
        }

        return reply;
    }

    /**
     * Get comments with hierarchical structure and flattened replies
     */
    public Page<CommentResponse> getCommentsByPost(String postId, Pageable pageable) {
        // Get root comments first
        Page<CommentEntity> rootComments = commentRepository.findRootCommentsByPostId(postId, pageable);

        return rootComments.map(rootComment -> {
            CommentResponse response = new CommentResponse(rootComment);

            // Load nested replies (up to max depth - 1)
            loadNestedReplies(response, CommentEntity.MAX_DEPTH - 1);

            // Load flattened replies
            List<CommentEntity> flattenedReplies = commentRepository.findFlattenedReplies(
                rootComment.getId(), CommentEntity.MAX_DEPTH
            );

            for (CommentEntity flattened : flattenedReplies) {
                response.addReply(new CommentResponse(flattened));
            }

            // Set total reply count
            response.setReplyCount(commentRepository.countTotalReplies(rootComment.getId()));

            return response;
        });
    }

    /**
     * Recursively load nested replies up to max depth
     */
    private void loadNestedReplies(CommentResponse parent, int maxDepth) {
        if (maxDepth <= 0) return;

        List<CommentEntity> directReplies = commentRepository.findDirectReplies(parent.getId(), CommentEntity.MAX_DEPTH);

        for (CommentEntity reply : directReplies) {
            CommentResponse replyResponse = new CommentResponse(reply);
            parent.addReply(replyResponse);

            // Recursively load deeper replies
            loadNestedReplies(replyResponse, maxDepth - 1);
        }
    }

    /**
     * Get all replies for a specific comment (both nested and flattened)
     */
    public List<CommentResponse> getReplies(String commentId) {
        List<CommentResponse> allReplies = new ArrayList<>();

        // Get direct nested replies
        List<CommentEntity> directReplies = commentRepository.findByParentCommentId(commentId);
        for (CommentEntity reply : directReplies) {
            CommentResponse replyResponse = new CommentResponse(reply);
            loadNestedReplies(replyResponse, CommentEntity.MAX_DEPTH - (reply.getDepth() != null ? reply.getDepth() : 0));
            allReplies.add(replyResponse);
        }

        // Get flattened replies
        List<CommentEntity> flattenedReplies = commentRepository.findFlattenedReplies(commentId, CommentEntity.MAX_DEPTH);
        for (CommentEntity flattened : flattenedReplies) {
            allReplies.add(new CommentResponse(flattened));
        }

        // Sort by creation time
        allReplies.sort((a, b) -> a.getCreatedAt().compareTo(b.getCreatedAt()));

        return allReplies;
    }

    public CommentResponse getCommentById(String id) {
        Optional<CommentEntity> commentOpt = commentRepository.findById(id);
        if (commentOpt.isPresent()) {
            CommentResponse response = new CommentResponse(commentOpt.get());

            // Load replies if it's a root comment
            if (response.isRootComment()) {
                loadNestedReplies(response, CommentEntity.MAX_DEPTH - 1);

                List<CommentEntity> flattenedReplies = commentRepository.findFlattenedReplies(
                    response.getId(), CommentEntity.MAX_DEPTH
                );
                for (CommentEntity flattened : flattenedReplies) {
                    response.addReply(new CommentResponse(flattened));
                }

                response.setReplyCount(commentRepository.countTotalReplies(response.getId()));
            }

            return response;
        }
        throw new RuntimeException("Comment not found with id: " + id);
    }

    public void deleteComment(String id, String authorId) {
        Optional<CommentEntity> commentOpt = commentRepository.findById(id);
        if (commentOpt.isPresent()) {
            CommentEntity comment = commentOpt.get();

            // Check if user is the author
            if (!comment.getAuthor().getId().equals(authorId)) {
                throw new RuntimeException("Only the author can delete this comment");
            }

            // Delete all replies (both nested and flattened)
            commentRepository.deleteAllRepliesUnderComment(id);

            // Update post comment count (count all deleted comments)
            long deletedCount = commentRepository.countTotalReplies(id) + 1; // +1 for the comment itself
            Optional<PostEntity> postOpt = postRepository.findById(comment.getPostId());
            if (postOpt.isPresent()) {
                PostEntity post = postOpt.get();
                long currentCount = post.getStats().getComments();
                post.getStats().setComments(Math.max(0, currentCount - deletedCount));
                postRepository.save(post);
            }

            // Delete the comment itself
            commentRepository.deleteById(id);

            // Publish event
            eventService.publishCommentEvent("COMMENT_DELETED", comment.getPostId(), id, authorId);
        } else {
            throw new RuntimeException("Comment not found with id: " + id);
        }
    }

    public long getCommentCountByPost(String postId) {
        return commentRepository.countByPostId(postId);
    }
}
