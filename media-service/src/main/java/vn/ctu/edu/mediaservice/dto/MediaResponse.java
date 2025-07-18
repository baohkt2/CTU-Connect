package vn.ctu.edu.mediaservice.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import vn.ctu.edu.mediaservice.entity.Media;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MediaResponse {
    private Long id;
    private String fileName;
    private String originalFileName;
    private String cloudinaryUrl;
    private String cloudinaryPublicId;
    private String contentType;
    private Media.MediaType mediaType;
    private Long fileSize;
    private String description;
    private String uploadedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
