package com.ctuconnect.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MediaUploadRequest {
    @NotBlank(message = "Uploaded by is required")
    private String uploadedBy;

    private String description;
}
