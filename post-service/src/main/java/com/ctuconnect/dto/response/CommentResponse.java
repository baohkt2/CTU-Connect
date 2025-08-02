package com.ctuconnect.dto.response;

import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.entity.CommentEntity;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Data
public class CommentResponse {

    // Getters and Setters
    private String id;
    private String postId;
    private String content;
    private AuthorInfo author;
    private String parentCommentId;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Constructors
    public CommentResponse() {}

    public CommentResponse(CommentEntity comment) {
        this.id = comment.getId();
        this.postId = comment.getPostId();
        this.content = comment.getContent();
        this.author = comment.getAuthor();
        this.parentCommentId = comment.getParentCommentId();
        this.createdAt = comment.getCreatedAt();
        this.updatedAt = comment.getUpdatedAt();
    }

}
