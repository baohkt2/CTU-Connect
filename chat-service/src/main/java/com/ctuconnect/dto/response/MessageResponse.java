package com.ctuconnect.dto.response;

import com.ctuconnect.model.Message;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MessageResponse {
    private String id;
    private String conversationId;
    private String senderId;
    private String senderName;
    private String senderAvatar;
    private Message.MessageType type;
    private String content;
    private MessageAttachmentResponse attachment;
    private String replyToMessageId;
    private MessageResponse replyToMessage;
    private List<MessageReactionResponse> reactions;
    private Message.MessageStatus status;
    private List<String> readByUserIds;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime editedAt;
    private boolean isEdited;
    private boolean isDeleted;
}
