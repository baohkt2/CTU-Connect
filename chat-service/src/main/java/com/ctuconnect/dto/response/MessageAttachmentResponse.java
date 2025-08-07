package com.ctuconnect.dto.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MessageAttachmentResponse {
    private String fileName;
    private String fileUrl;
    private String fileType;
    private Long fileSize;
    private String thumbnailUrl;
}
