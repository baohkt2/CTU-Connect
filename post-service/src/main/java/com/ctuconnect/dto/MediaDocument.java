package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class MediaDocument {
    private String id;
    private String fileName;
    private String originalFileName;
    private String url;
    private String contentType;
    private long fileSize;
    private LocalDateTime uploadedAt;
}
