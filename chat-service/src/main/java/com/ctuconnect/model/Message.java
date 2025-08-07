package com.ctuconnect.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.index.Indexed;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "messages")
public class Message {
    @Id
    private String id;

    @Indexed
    private String conversationId; // ID của conversation

    @Indexed
    private String senderId; // ID người gửi

    private String senderName; // Tên người gửi (cache)

    private String senderAvatar; // Avatar người gửi (cache)

    private MessageType type; // TEXT, IMAGE, FILE, SYSTEM

    private String content; // Nội dung tin nhắn

    private MessageAttachment attachment; // File đính kèm

    private String replyToMessageId; // ID tin nhắn được reply

    private List<MessageReaction> reactions = new ArrayList<>(); // Reactions

    private MessageStatus status; // SENT, DELIVERED, READ

    private List<String> readByUserIds = new ArrayList<>(); // Danh sách user đã đọc

    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;

    private LocalDateTime editedAt;

    private boolean isEdited = false; // Có phải là tin nhắn đã chỉnh sửa

    private boolean isDeleted = false;

    public enum MessageType {
        TEXT,
        IMAGE,
        FILE,
        SYSTEM // Tin nhắn hệ thống (join, leave, etc.)
    }

    public enum MessageStatus {
        SENT,
        DELIVERED,
        READ
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MessageAttachment {
        private String fileName;
        private String fileUrl;
        private String fileType; // image/jpeg, application/pdf, etc.
        private Long fileSize;
        private String thumbnailUrl; // Cho images/videos
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class MessageReaction {
        private String userId;
        private String userName;
        private String emoji; // 👍, ❤️, 😂, 😮, 😢, 😡
        private LocalDateTime createdAt;
    }
}
