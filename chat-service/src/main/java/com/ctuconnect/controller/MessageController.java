package com.ctuconnect.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import com.ctuconnect.dto.request.AddReactionRequest;
import com.ctuconnect.dto.request.SendMessageRequest;
import com.ctuconnect.dto.response.ChatPageResponse;
import com.ctuconnect.dto.response.MessageResponse;
import com.ctuconnect.service.MessageService;

import jakarta.validation.Valid;
import java.util.List;

@RestController
@RequestMapping("/api/messages")
@RequiredArgsConstructor
public class MessageController {

    private final MessageService messageService;

    @PostMapping
    public ResponseEntity<MessageResponse> sendMessage(
            @Valid @RequestBody SendMessageRequest request,
            Authentication authentication) {

        String userId = authentication.getName();
        MessageResponse response = messageService.sendMessage(request, userId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/conversation/{conversationId}")
    public ResponseEntity<ChatPageResponse<MessageResponse>> getMessages(
            @PathVariable String conversationId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "50") int size,
            Authentication authentication) {

        String userId = authentication.getName();
        ChatPageResponse<MessageResponse> messages = messageService.getMessages(conversationId, userId, page, size);
        return ResponseEntity.ok(messages);
    }

    @PutMapping("/{messageId}")
    public ResponseEntity<MessageResponse> editMessage(
            @PathVariable String messageId,
            @RequestParam String content,
            Authentication authentication) {

        String userId = authentication.getName();
        MessageResponse response = messageService.editMessage(messageId, content, userId);
        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/{messageId}")
    public ResponseEntity<Void> deleteMessage(
            @PathVariable String messageId,
            Authentication authentication) {

        String userId = authentication.getName();
        messageService.deleteMessage(messageId, userId);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/reactions")
    public ResponseEntity<MessageResponse> addReaction(
            @Valid @RequestBody AddReactionRequest request,
            Authentication authentication) {

        String userId = authentication.getName();
        MessageResponse response = messageService.addReaction(request, userId);
        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/{messageId}/reactions")
    public ResponseEntity<Void> removeReaction(
            @PathVariable String messageId,
            Authentication authentication) {

        String userId = authentication.getName();
        messageService.removeReaction(messageId, userId);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/conversation/{conversationId}/search")
    public ResponseEntity<List<MessageResponse>> searchMessages(
            @PathVariable String conversationId,
            @RequestParam String query,
            Authentication authentication) {

        String userId = authentication.getName();
        List<MessageResponse> results = messageService.searchMessages(conversationId, query, userId);
        return ResponseEntity.ok(results);
    }

    @GetMapping("/conversation/{conversationId}/unread-count")
    public ResponseEntity<Long> getUnreadCount(
            @PathVariable String conversationId,
            Authentication authentication) {

        String userId = authentication.getName();
        long count = messageService.getUnreadCount(conversationId, userId);
        return ResponseEntity.ok(count);
    }

    @PostMapping("/conversation/{conversationId}/mark-read")
    public ResponseEntity<Void> markAsRead(
            @PathVariable String conversationId,
            Authentication authentication) {

        String userId = authentication.getName();
        messageService.markMessagesAsRead(conversationId, userId);
        return ResponseEntity.ok().build();
    }

    // WebSocket message mappings for real-time messaging
    @MessageMapping("/message.send")
    public void sendMessageViaWebSocket(@Payload SendMessageRequest request, Authentication authentication) {
        String userId = authentication.getName();
        messageService.sendMessage(request, userId);
    }

    @MessageMapping("/message.reaction")
    public void addReactionViaWebSocket(@Payload AddReactionRequest request, Authentication authentication) {
        String userId = authentication.getName();
        messageService.addReaction(request, userId);
    }
}
