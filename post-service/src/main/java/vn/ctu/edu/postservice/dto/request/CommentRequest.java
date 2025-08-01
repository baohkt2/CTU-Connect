package vn.ctu.edu.postservice.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class CommentRequest {

    @NotBlank(message = "Content is required")
    @Size(max = 1000, message = "Content cannot exceed 1000 characters")
    private String content;

    private String parentCommentId; // Optional for replies
}
