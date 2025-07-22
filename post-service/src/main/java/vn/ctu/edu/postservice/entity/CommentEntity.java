package vn.ctu.edu.postservice.entity;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDateTime;

@Document(collection = "comments")
public class CommentEntity {

    @Id
    private String id;

    @Field("post_id")
    private String postId;

    private String content;

    @Field("author_id")
    private String authorId;

    @Field("parent_comment_id")
    private String parentCommentId; // For nested comments/replies

    @Field("created_at")
    private LocalDateTime createdAt;

    @Field("updated_at")
    private LocalDateTime updatedAt;

    // Constructors
    public CommentEntity() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public CommentEntity(String postId, String content, String authorId) {
        this();
        this.postId = postId;
        this.content = content;
        this.authorId = authorId;
    }

    public CommentEntity(String postId, String content, String authorId, String parentCommentId) {
        this(postId, content, authorId);
        this.parentCommentId = parentCommentId;
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

    public void updateTimestamp() {
        this.updatedAt = LocalDateTime.now();
    }
}
