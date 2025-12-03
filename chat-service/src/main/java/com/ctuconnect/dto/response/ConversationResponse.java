package com.ctuconnect.dto.response;

import com.ctuconnect.model.Conversation;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ConversationResponse {
    private String id;
    private String name;
    private Conversation.ConversationType type;
    private List<ParticipantInfo> participants;
    private MessageResponse lastMessage;
    private LocalDateTime lastMessageAt;
    private int unreadCount;
    private String avatarUrl;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
