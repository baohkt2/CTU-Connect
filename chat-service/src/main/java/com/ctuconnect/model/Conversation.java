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
@Document(collection = "conversations")
public class Conversation {
    @Id
    private String id;
    
    @Indexed
    private String name; // Tên nhóm chat (null nếu là chat 1-1)
    
    @Indexed
    private ConversationType type; // DIRECT, GROUP
    
    private List<String> participantIds = new ArrayList<>(); // Danh sách user IDs
    
    private String lastMessageId; // ID tin nhắn cuối cùng
    
    private LocalDateTime lastMessageAt; // Thời gian tin nhắn cuối
    
    private String createdBy; // Người tạo conversation
    
    private LocalDateTime createdAt;
    
    private LocalDateTime updatedAt;
    
    // Metadata cho group chat
    private ConversationMetadata metadata;
    
    // Settings
    private ConversationSettings settings;
    
    public enum ConversationType {
        DIRECT, // Chat 1-1
        GROUP   // Chat nhóm
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ConversationMetadata {
        private String description; // Mô tả nhóm
        private String avatarUrl;   // Avatar nhóm
        private List<String> adminIds = new ArrayList<>(); // Danh sách admin
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ConversationSettings {
        private boolean allowMembersToAddOthers = true;
        private boolean allowMembersToChangeInfo = false;
        private boolean muteNotifications = false;
    }
}
