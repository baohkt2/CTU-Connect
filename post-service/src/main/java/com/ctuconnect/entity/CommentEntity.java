package com.ctuconnect.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.mapping.Document;

import org.springframework.data.mongodb.core.mapping.Field;
import lombok.Data;
import com.ctuconnect.dto.AuthorInfo;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@Builder
@Document(collection = "comments")
public class CommentEntity {

    @Id
    private String id;

    @Field("post_id")
    private String postId;

    private String content;

    @Field("author")
    private AuthorInfo author;

    @Field("parent_comment_id")
    private String parentCommentId; // For nested comments/replies

    @Field("created_at")
    @CreatedDate
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Field("updated_at")
    private LocalDateTime updatedAt;

    // Constructors
    public CommentEntity() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public CommentEntity(String postId, String content, AuthorInfo author) {
        this();
        this.postId = postId;
        this.content = content;
        this.author = author;
    }

    public CommentEntity(String postId, String content, AuthorInfo author, String parentCommentId) {
        this(postId, content, author);
        this.parentCommentId = parentCommentId;
    }

    public String getAuthorId() {
        return author != null ? author.getId() : null;
    }
}
