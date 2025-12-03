package com.ctuconnect.controller;

import com.ctuconnect.dto.request.AddReactionRequest;
import com.ctuconnect.dto.request.SendMessageRequest;
import com.ctuconnect.dto.request.TypingRequest;
import com.ctuconnect.dto.response.ChatPageResponse;
import com.ctuconnect.dto.response.MessageResponse;
import com.ctuconnect.security.SecurityContextHolder;
import com.ctuconnect.service.MessageService;
import com.ctuconnect.service.UserPresenceService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessageHeaderAccessor;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.security.Principal;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/chats/messages")
@RequiredArgsConstructor
@Slf4j
public class MessageController {

    private final MessageService messageService;
    private final UserPresenceService userPresenceService;
    private final SimpMessagingTemplate messagingTemplate;

    // HTTP REST endpoints (existing code)
    @PostMapping
    public ResponseEntity<MessageResponse> sendMessage(
            @Valid @RequestBody SendMessageRequest request) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        MessageResponse response = messageService.sendMessage(request, userId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/conversation/{conversationId}")
    public ResponseEntity<ChatPageResponse<MessageResponse>> getMessages(
            @PathVariable String conversationId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "50") int size) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        ChatPageResponse<MessageResponse> messages = messageService.getMessages(conversationId, userId, page, size);
        return ResponseEntity.ok(messages);
    }

    @PutMapping("/{messageId}")
    public ResponseEntity<MessageResponse> editMessage(
            @PathVariable String messageId,
            @RequestParam String content) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        MessageResponse response = messageService.editMessage(messageId, content, userId);
        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/{messageId}")
    public ResponseEntity<Void> deleteMessage(
            @PathVariable String messageId) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        messageService.deleteMessage(messageId, userId);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/reactions")
    public ResponseEntity<MessageResponse> addReaction(
            @Valid @RequestBody AddReactionRequest request) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        MessageResponse response = messageService.addReaction(request, userId);
        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/{messageId}/reactions")
    public ResponseEntity<Void> removeReaction(
            @PathVariable String messageId) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        messageService.removeReaction(messageId, userId);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/conversation/{conversationId}/search")
    public ResponseEntity<List<MessageResponse>> searchMessages(
            @PathVariable String conversationId,
            @RequestParam String query) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        List<MessageResponse> results = messageService.searchMessages(conversationId, query, userId);
        return ResponseEntity.ok(results);
    }

    @GetMapping("/conversation/{conversationId}/unread-count")
    public ResponseEntity<Long> getUnreadCount(
            @PathVariable String conversationId) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        long count = messageService.getUnreadCount(conversationId, userId);
        return ResponseEntity.ok(count);
    }

    @PostMapping("/conversation/{conversationId}/mark-read")
    public ResponseEntity<Void> markAsRead(
            @PathVariable String conversationId) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        messageService.markMessagesAsRead(conversationId, userId);
        return ResponseEntity.ok().build();
    }

    // WebSocket STOMP endpoints for real-time communication

    /**
     * Handle real-time message sending via WebSocket
     */
    @MessageMapping("/chat.sendMessage")
    public void handleWebSocketMessage(
            @Payload SendMessageRequest request,
            SimpMessageHeaderAccessor headerAccessor,
            Principal principal) {

        try {
            String userId = principal.getName();
            log.debug("Received WebSocket message from user: {}", userId);

            // Send message through service
            MessageResponse response = messageService.sendMessage(request, userId);

            // Broadcast to conversation participants
            messagingTemplate.convertAndSend(
                    "/topic/conversations/" + request.getConversationId(),
                    response
            );

            // Send confirmation to sender
            messagingTemplate.convertAndSendToUser(
                    userId,
                    "/queue/message-sent",
                    Map.of("messageId", response.getId(), "status", "sent")
            );

        } catch (Exception e) {
            log.error("Error handling WebSocket message", e);

            // Send error to sender
            if (principal != null) {
                messagingTemplate.convertAndSendToUser(
                        principal.getName(),
                        "/queue/errors",
                        Map.of("error", e.getMessage(), "type", "MESSAGE_SEND_FAILED")
                );
            }
        }
    }

    /**
     * Handle typing indicators via WebSocket
     */
    @MessageMapping("/chat.typing")
    public void handleTyping(
            @Payload TypingRequest request,
            Principal principal) {

        try {
            String userId = principal.getName();
            log.debug("Received typing indicator from user: {} for conversation: {}",
                    userId, request.getConversationId());

            // Broadcast typing status to other participants (excluding sender)
            messagingTemplate.convertAndSend(
                    "/topic/conversations/" + request.getConversationId() + "/typing",
                    Map.of(
                            "userId", userId,
                            "isTyping", request.isTyping(),
                            "timestamp", System.currentTimeMillis()
                    )
            );

        } catch (Exception e) {
            log.error("Error handling typing indicator", e);
        }
    }

    /**
     * Handle user presence updates via WebSocket
     */
    @MessageMapping("/user.presence")
    public void handlePresenceUpdate(
            @Payload Map<String, Object> presenceData,
            Principal principal) {

        try {
            String userId = principal.getName();
            String status = (String) presenceData.get("status");

            log.debug("Updating presence for user: {} to status: {}", userId, status);

            // Update user presence
            userPresenceService.updateUserPresence(userId, status);

            // Broadcast presence update
            messagingTemplate.convertAndSend(
                    "/topic/presence",
                    Map.of(
                            "userId", userId,
                            "status", status,
                            "timestamp", System.currentTimeMillis()
                    )
            );

        } catch (Exception e) {
            log.error("Error updating user presence", e);
        }
    }

    /**
     * Handle conversation join events
     */
    @MessageMapping("/chat.join")
    public void handleJoinConversation(
            @Payload Map<String, String> joinData,
            Principal principal) {

        try {
            String userId = principal.getName();
            String conversationId = joinData.get("conversationId");

            log.debug("User {} joining conversation: {}", userId, conversationId);

            // Update user's active conversation
            userPresenceService.updateActiveConversation(userId, conversationId);

            // Send join confirmation
            messagingTemplate.convertAndSendToUser(
                    userId,
                    "/queue/conversation-joined",
                    Map.of("conversationId", conversationId, "status", "joined")
            );

        } catch (Exception e) {
            log.error("Error handling conversation join", e);
        }
    }

    /**
     * Handle conversation leave events
     */
    @MessageMapping("/chat.leave")
    public void handleLeaveConversation(
            @Payload Map<String, String> leaveData,
            Principal principal) {

        try {
            String userId = principal.getName();
            String conversationId = leaveData.get("conversationId");

            log.debug("User {} leaving conversation: {}", userId, conversationId);

            // Clear user's active conversation
            userPresenceService.clearActiveConversation(userId);

            // Send leave confirmation
            messagingTemplate.convertAndSendToUser(
                    userId,
                    "/queue/conversation-left",
                    Map.of("conversationId", conversationId, "status", "left")
            );

        } catch (Exception e) {
            log.error("Error handling conversation leave", e);
        }
    }
}
