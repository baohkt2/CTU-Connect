package vn.ctu.edu.postservice.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import vn.ctu.edu.postservice.dto.request.CreateCommentRequest;
import vn.ctu.edu.postservice.dto.response.CommentResponse;
import vn.ctu.edu.postservice.entity.CommentEntity;
import vn.ctu.edu.postservice.entity.PostEntity;
import vn.ctu.edu.postservice.repository.CommentRepository;
import vn.ctu.edu.postservice.repository.PostRepository;

import java.util.Optional;

@Service
public class CommentService {

    @Autowired
    private CommentRepository commentRepository;

    @Autowired
    private PostRepository postRepository;

    @Autowired
    private EventService eventService;

    public CommentResponse createComment(String postId, CreateCommentRequest request) {
        // Verify post exists
        Optional<PostEntity> postOpt = postRepository.findById(postId);
        if (postOpt.isEmpty()) {
            throw new RuntimeException("Post not found with id: " + postId);
        }

        CommentEntity comment = new CommentEntity(postId, request.getContent(), request.getAuthorId(), request.getParentCommentId());
        CommentEntity savedComment = commentRepository.save(comment);

        // Update post comment count
        PostEntity post = postOpt.get();
        post.getStats().incrementComments();
        postRepository.save(post);

        // Publish event
        eventService.publishCommentEvent("COMMENT_CREATED", postId, savedComment.getId(), savedComment.getAuthorId());

        return new CommentResponse(savedComment);
    }

    public Page<CommentResponse> getCommentsByPost(String postId, Pageable pageable) {
        return commentRepository.findByPostId(postId, pageable)
                .map(CommentResponse::new);
    }

    public CommentResponse getCommentById(String id) {
        Optional<CommentEntity> commentOpt = commentRepository.findById(id);
        if (commentOpt.isPresent()) {
            return new CommentResponse(commentOpt.get());
        }
        throw new RuntimeException("Comment not found with id: " + id);
    }

    public void deleteComment(String id, String authorId) {
        Optional<CommentEntity> commentOpt = commentRepository.findById(id);
        if (commentOpt.isPresent()) {
            CommentEntity comment = commentOpt.get();

            // Check if user is the author
            if (!comment.getAuthorId().equals(authorId)) {
                throw new RuntimeException("Only the author can delete this comment");
            }

            // Update post comment count
            Optional<PostEntity> postOpt = postRepository.findById(comment.getPostId());
            if (postOpt.isPresent()) {
                PostEntity post = postOpt.get();
                post.getStats().decrementComments();
                postRepository.save(post);
            }

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
