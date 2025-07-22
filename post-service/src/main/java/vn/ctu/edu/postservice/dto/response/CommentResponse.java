package vn.ctu.edu.postservice.dto.response;

import vn.ctu.edu.postservice.entity.CommentEntity;

import java.time.LocalDateTime;

public class CommentResponse {

    private String id;
    private String postId;
    private String content;
    private String authorId;
    private String parentCommentId;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Constructors
    public CommentResponse() {}

    public CommentResponse(CommentEntity comment) {
        this.id = comment.getId();
        this.postId = comment.getPostId();
        this.content = comment.getContent();
        this.authorId = comment.getAuthorId();
        this.parentCommentId = comment.getParentCommentId();
        this.createdAt = comment.getCreatedAt();
        this.updatedAt = comment.getUpdatedAt();
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getPostId() {
        return postId;
    }

    public void setPostId(String postId) {
        this.postId = postId;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getAuthorId() {
        return authorId;
    }

    public void setAuthorId(String authorId) {
        this.authorId = authorId;
    }

    public String getParentCommentId() {
        return parentCommentId;
    }

    public void setParentCommentId(String parentCommentId) {
        this.parentCommentId = parentCommentId;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
}
