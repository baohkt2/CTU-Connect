package com.ctuconnect.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.dto.MediaDocument;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class PostRequest {
    @Size(max = 200, message = "Title cannot exceed 200 characters")
    private String title;

    @NotBlank(message = "Content is required")
    @Size(max = 5000, message = "Content cannot exceed 5000 characters")
    private String content;

    @Builder.Default
    private List<String> tags = new ArrayList<>();

    private String category;

    private String visibility = "PUBLIC"; // PUBLIC, FRIENDS, PRIVATE
    
    // Enhanced fields for Facebook-like functionality
    @Builder.Default
    private List<String> images = new ArrayList<>();
    
    @Builder.Default
    private List<String> videos = new ArrayList<>();
    
    @Builder.Default
    private List<MediaDocument> documents = new ArrayList<>();

    private String postType; // TEXT, IMAGE, VIDEO, LINK, POLL, EVENT, SHARED
    
    private PostEntity.AudienceSettings audienceSettings;
    
    private LocalDateTime scheduledAt;
    
    private PostEntity.LocationInfo location;
}
