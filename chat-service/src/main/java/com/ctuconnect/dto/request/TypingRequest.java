package com.ctuconnect.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.validation.constraints.NotBlank;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TypingRequest {
    @NotBlank(message = "ID conversation không được trống")
    private String conversationId;

    private boolean isTyping;
}
