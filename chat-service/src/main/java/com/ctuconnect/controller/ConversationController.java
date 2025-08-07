package com.ctuconnect.controller;

import com.ctuconnect.dto.request.CreateConversationRequest;
import com.ctuconnect.dto.response.ConversationResponse;
import com.ctuconnect.security.SecurityContextHolder;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.http.ResponseEntity;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import com.ctuconnect.dto.request.*;
import com.ctuconnect.dto.response.*;
import com.ctuconnect.service.ConversationService;
import com.ctuconnect.service.UserPresenceService;

import jakarta.validation.Valid;
import java.util.List;

@RestController
@RequestMapping("/api/chats/conversations")
@RequiredArgsConstructor
public class ConversationController {

    private final ConversationService conversationService;
    private final UserPresenceService userPresenceService;

    @PostMapping
    public ResponseEntity<ConversationResponse> createConversation(
            @Valid @RequestBody CreateConversationRequest request) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        ConversationResponse response = conversationService.createConversation(request, userId);
        return ResponseEntity.ok(response);
    }

    @GetMapping
    public ResponseEntity<Page<ConversationResponse>> getUserConversations(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        Page<ConversationResponse> conversations = conversationService.getUserConversations(userId, page, size);
        return ResponseEntity.ok(conversations);
    }

    @GetMapping("/{conversationId}")
    public ResponseEntity<ConversationResponse> getConversation(
            @PathVariable String conversationId) {
        
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        ConversationResponse response = conversationService.getConversationById(conversationId, userId);
        return ResponseEntity.ok(response);
    }
    
    @PutMapping("/{conversationId}")
    public ResponseEntity<ConversationResponse> updateConversation(
            @PathVariable String conversationId,
            @Valid @RequestBody UpdateConversationRequest request) {
        
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        ConversationResponse response = conversationService.updateConversation(conversationId, request, userId);
        return ResponseEntity.ok(response);
    }
    
    @PostMapping("/{conversationId}/participants")
    public ResponseEntity<Void> addParticipant(
            @PathVariable String conversationId,
            @RequestParam String participantId) {
        
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        conversationService.addParticipant(conversationId, participantId, userId);
        return ResponseEntity.ok().build();
    }
    
    @DeleteMapping("/{conversationId}/participants/{participantId}")
    public ResponseEntity<Void> removeParticipant(
            @PathVariable String conversationId,
            @PathVariable String participantId) {
        
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        conversationService.removeParticipant(conversationId, participantId, userId);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/search")
    public ResponseEntity<List<ConversationResponse>> searchConversations(
            @RequestParam String query) {
        
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        List<ConversationResponse> results = conversationService.searchConversations(userId, query);
        return ResponseEntity.ok(results);
    }
    
    @DeleteMapping("/{conversationId}")
    public ResponseEntity<Void> deleteConversation(
            @PathVariable String conversationId) {
        
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        conversationService.deleteConversation(conversationId, userId);
        return ResponseEntity.noContent().build();
    }
    
    // WebSocket message mappings for real-time features
    @MessageMapping("/conversation/{conversationId}/typing")
    public void handleTyping(@Payload TypingRequest request, Authentication authentication) {
        String userId = SecurityContextHolder.getCurrentUserIdOrThrow();
        userPresenceService.setTypingStatus(userId, request.getConversationId(), request.isTyping());
    }
}
