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

    @Field("root_comment_id")
    private String rootCommentId; // For flattened comments beyond max depth

    @Field("depth")
    private Integer depth; // Comment nesting depth (0 = root comment)

    @Field("reply_to_author")
    private String replyToAuthor; // Name of author being replied to (for flattened comments)

    @Field("created_at")
    @CreatedDate
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Field("updated_at")
    private LocalDateTime updatedAt;

    // Constants
    public static final int MAX_DEPTH = 3;

    // Constructors
    public CommentEntity() {
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
        this.depth = 0;
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

    // Enhanced constructor for reply with depth management
    public CommentEntity(String postId, String content, AuthorInfo author,
                        String parentCommentId, String rootCommentId,
                        Integer depth, String replyToAuthor) {
        this(postId, content, author);
        this.parentCommentId = parentCommentId;
        this.rootCommentId = rootCommentId;
        this.depth = depth != null ? depth : 0;
        this.replyToAuthor = replyToAuthor;
    }

    public String getAuthorId() {
        return author != null ? author.getId() : null;
    }

    // Helper methods for depth management
    public boolean isRootComment() {
        return depth == null || depth == 0;
    }

    public boolean shouldFlatten() {
        return depth != null && depth >= MAX_DEPTH;
    }

    public String getDisplayContent() {
        if (replyToAuthor != null && !replyToAuthor.isEmpty()) {
            return "@" + replyToAuthor + " " + content;
        }
        return content;
    }
}
