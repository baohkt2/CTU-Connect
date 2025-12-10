package com.ctuconnect.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SendMessageRequest {
    @NotBlank(message = "ID conversation không được trống")
    private String conversationId;

    @Size(max = 2000, message = "Tin nhắn không được vượt quá 2000 ký tự")
    private String content; // Content có thể null nếu có attachment

    private String replyToMessageId;
    
    private MessageAttachmentRequest attachment;
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MessageAttachmentRequest {
        private String fileName;
        private String fileUrl;
        private String fileType;
        private Long fileSize;
        private String thumbnailUrl;
    }
}
