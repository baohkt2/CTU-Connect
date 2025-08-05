package com.ctuconnect.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import com.ctuconnect.model.Conversation;
import com.ctuconnect.model.Message;

import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationService {
    
    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final WebSocketService webSocketService;
    
    private static final String NOTIFICATION_TOPIC = "chat_notifications";
    
    public void sendMessageNotification(Conversation conversation, Message message) {
        // Tạo notification event
        Map<String, Object> notificationEvent = new HashMap<>();
        notificationEvent.put("type", "NEW_MESSAGE");
        notificationEvent.put("conversationId", conversation.getId());
        notificationEvent.put("messageId", message.getId());
        notificationEvent.put("senderId", message.getSenderId());
        notificationEvent.put("senderName", message.getSenderName());
        notificationEvent.put("content", message.getContent());
        notificationEvent.put("conversationName", conversation.getName());
        notificationEvent.put("conversationType", conversation.getType().toString());
        
        // Gửi notification cho từng participant (trừ sender)
        conversation.getParticipantIds().stream()
            .filter(participantId -> !participantId.equals(message.getSenderId()))
            .forEach(participantId -> {
                notificationEvent.put("recipientId", participantId);
                
                // Gửi qua Kafka để notification service xử lý
                kafkaTemplate.send(NOTIFICATION_TOPIC, participantId, notificationEvent);
                
                // Gửi real-time notification qua WebSocket
                webSocketService.sendNotificationToUser(participantId, notificationEvent);
                
                log.debug("Sent message notification to user: {}", participantId);
            });
    }
    
    public void sendConversationUpdateNotification(Conversation conversation, String updateType, String updatedBy) {
        Map<String, Object> notificationEvent = new HashMap<>();
        notificationEvent.put("type", "CONVERSATION_UPDATE");
        notificationEvent.put("conversationId", conversation.getId());
        notificationEvent.put("updateType", updateType); // MEMBER_ADDED, MEMBER_REMOVED, INFO_UPDATED
        notificationEvent.put("updatedBy", updatedBy);
        notificationEvent.put("conversationName", conversation.getName());
        
        // Gửi cho tất cả participants
        conversation.getParticipantIds().forEach(participantId -> {
            notificationEvent.put("recipientId", participantId);
            kafkaTemplate.send(NOTIFICATION_TOPIC, participantId, notificationEvent);
            webSocketService.sendNotificationToUser(participantId, notificationEvent);
        });
    }
}
