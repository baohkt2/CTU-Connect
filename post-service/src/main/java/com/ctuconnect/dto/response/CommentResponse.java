package com.ctuconnect.dto.response;

import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.entity.CommentEntity;
import lombok.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.ArrayList;

@Data
@Builder
@AllArgsConstructor

public class CommentResponse {

    // Getters and Setters
    private String id;
    private String postId;
    private String content;
    private AuthorInfo author;
    private String parentCommentId;
    private String rootCommentId;
    private Integer depth;
    private String replyToAuthor;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Additional fields for UI
    private List<CommentResponse> replies;
    private Long replyCount;
    private boolean isFlattened;

    // Constructors
    public CommentResponse() {
        this.replies = new ArrayList<>();
        this.replyCount = 0L;
    }

    public CommentResponse(CommentEntity comment) {
        this.id = comment.getId();
        this.postId = comment.getPostId();
        this.content = comment.getContent();
        this.author = comment.getAuthor();
        this.parentCommentId = comment.getParentCommentId();
        this.rootCommentId = comment.getRootCommentId();
        this.depth = comment.getDepth();
        this.replyToAuthor = comment.getReplyToAuthor();
        this.createdAt = comment.getCreatedAt();
        this.updatedAt = comment.getUpdatedAt();
        this.replies = new ArrayList<>();
        this.replyCount = 0L;
        this.isFlattened = comment.shouldFlatten();
    }

    // Helper methods
    public String getDisplayContent() {
        if (replyToAuthor != null && !replyToAuthor.isEmpty()) {
            return "@" + replyToAuthor + " " + content;
        }
        return content;
    }

    public boolean isRootComment() {
        return depth == null || depth == 0;
    }

    public boolean shouldShowReplyButton() {
        return depth == null || depth < CommentEntity.MAX_DEPTH;
    }

    public void addReply(CommentResponse reply) {
        this.replies.add(reply);
        this.replyCount = (long) this.replies.size();
    }
}
