package com.ctuconnect.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;
import com.ctuconnect.dto.response.MessageResponse;
import com.ctuconnect.dto.response.UserPresenceResponse;

@Service
@RequiredArgsConstructor
@Slf4j
public class WebSocketService {
    
    private final SimpMessagingTemplate messagingTemplate;
    
    // Send message to all users in a conversation
    public void sendMessageToConversation(String conversationId, MessageResponse message) {
        String destination = "/topic/conversation/" + conversationId + "/messages";
        messagingTemplate.convertAndSend(destination, message);
        log.debug("Sent message to conversation {}: {}", conversationId, message.getId());
    }
    
    // Send message update (edit) to conversation
    public void sendMessageUpdateToConversation(String conversationId, MessageResponse message) {
        String destination = "/topic/conversation/" + conversationId + "/messages/update";
        messagingTemplate.convertAndSend(destination, message);
        log.debug("Sent message update to conversation {}: {}", conversationId, message.getId());
    }
    
    // Send message delete notification
    public void sendMessageDeleteToConversation(String conversationId, String messageId) {
        String destination = "/topic/conversation/" + conversationId + "/messages/delete";
        messagingTemplate.convertAndSend(destination, messageId);
        log.debug("Sent message delete to conversation {}: {}", conversationId, messageId);
    }
    
    // Send reaction update
    public void sendReactionUpdateToConversation(String conversationId, MessageResponse message) {
        String destination = "/topic/conversation/" + conversationId + "/reactions";
        messagingTemplate.convertAndSend(destination, message);
        log.debug("Sent reaction update to conversation {}: {}", conversationId, message.getId());
    }
    
    // Send typing status
    public void broadcastTypingStatus(String conversationId, String userId, boolean isTyping) {
        String destination = "/topic/conversation/" + conversationId + "/typing";
        TypingEvent event = new TypingEvent(userId, isTyping);
        messagingTemplate.convertAndSend(destination, event);
        log.debug("Broadcasted typing status for user {} in conversation {}: {}", userId, conversationId, isTyping);
    }
    
    // Send read receipt
    public void sendReadReceiptToConversation(String conversationId, String userId) {
        String destination = "/topic/conversation/" + conversationId + "/read";
        ReadReceiptEvent event = new ReadReceiptEvent(userId);
        messagingTemplate.convertAndSend(destination, event);
        log.debug("Sent read receipt for user {} in conversation {}", userId, conversationId);
    }
    
    // Broadcast user presence update
    public void broadcastPresenceUpdate(UserPresenceResponse presence) {
        String destination = "/topic/presence";
        messagingTemplate.convertAndSend(destination, presence);
        log.debug("Broadcasted presence update for user {}: {}", presence.getUserId(), presence.getStatus());
    }
    
    // Send notification to specific user
    public void sendNotificationToUser(String userId, Object notification) {
        String destination = "/queue/user/" + userId + "/notifications";
        messagingTemplate.convertAndSend(destination, notification);
        log.debug("Sent notification to user {}", userId);
    }
    
    // Send conversation update (new conversation, member added/removed)
    public void sendConversationUpdate(String conversationId, Object update) {
        String destination = "/topic/conversation/" + conversationId + "/updates";
        messagingTemplate.convertAndSend(destination, update);
        log.debug("Sent conversation update to {}", conversationId);
    }
    
    // Event classes
    public static class TypingEvent {
        private String userId;
        private boolean isTyping;
        
        public TypingEvent(String userId, boolean isTyping) {
            this.userId = userId;
            this.isTyping = isTyping;
        }
        
        // Getters and setters
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        public boolean isTyping() { return isTyping; }
        public void setTyping(boolean typing) { isTyping = typing; }
    }
    
    public static class ReadReceiptEvent {
        private String userId;
        
        public ReadReceiptEvent(String userId) {
            this.userId = userId;
        }
        
        // Getters and setters
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
    }
}
