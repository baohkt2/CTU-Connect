package com.ctuconnect.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class TypingRequest {

    @NotBlank(message = "Conversation ID cannot be blank")
    private String conversationId;

    @NotNull(message = "Typing status cannot be null")
    private boolean isTyping;

    private Long timestamp;

    public TypingRequest() {
        this.timestamp = System.currentTimeMillis();
    }

    public TypingRequest(String conversationId, boolean isTyping) {
        this.conversationId = conversationId;
        this.isTyping = isTyping;
        this.timestamp = System.currentTimeMillis();
    }
}
