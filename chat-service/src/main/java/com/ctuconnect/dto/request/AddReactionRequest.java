package com.ctuconnect.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import jakarta.validation.constraints.NotBlank;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AddReactionRequest {
    @NotBlank(message = "ID tin nhắn không được trống")
    private String messageId;

    @NotBlank(message = "Emoji không được trống")
    private String emoji;
}
