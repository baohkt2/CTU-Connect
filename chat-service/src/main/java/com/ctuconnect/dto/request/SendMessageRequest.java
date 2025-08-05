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

    @NotBlank(message = "Nội dung tin nhắn không được trống")
    @Size(max = 2000, message = "Tin nhắn không được vượt quá 2000 ký tự")
    private String content;

    private String replyToMessageId;
}
