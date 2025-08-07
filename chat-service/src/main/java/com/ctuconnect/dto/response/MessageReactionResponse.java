package com.ctuconnect.dto.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MessageReactionResponse {
    private String userId;
    private String userName;
    private String emoji;
    private LocalDateTime createdAt;
}
