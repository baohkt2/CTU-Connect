package com.ctuconnect.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import com.ctuconnect.dto.response.UserPresenceResponse;
import com.ctuconnect.service.UserPresenceService;

import java.util.List;

@RestController
@RequestMapping("/api/presence")
@RequiredArgsConstructor
public class UserPresenceController {

    private final UserPresenceService userPresenceService;

    @GetMapping("/{userId}")
    public ResponseEntity<UserPresenceResponse> getUserPresence(@PathVariable String userId) {
        UserPresenceResponse presence = userPresenceService.getUserPresence(userId);
        return ResponseEntity.ok(presence);
    }

    @GetMapping("/users")
    public ResponseEntity<List<UserPresenceResponse>> getMultipleUserPresence(
            @RequestParam List<String> userIds) {
        List<UserPresenceResponse> presences = userPresenceService.getMultipleUserPresence(userIds);
        return ResponseEntity.ok(presences);
    }

    @GetMapping("/online")
    public ResponseEntity<List<UserPresenceResponse>> getOnlineUsers() {
        List<UserPresenceResponse> onlineUsers = userPresenceService.getOnlineUsers();
        return ResponseEntity.ok(onlineUsers);
    }

    @PostMapping("/away")
    public ResponseEntity<Void> setAway(Authentication authentication) {
        String userId = authentication.getName();
        userPresenceService.setUserAway(userId);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/conversation/{conversationId}/typing")
    public ResponseEntity<List<String>> getTypingUsers(@PathVariable String conversationId) {
        List<String> typingUsers = userPresenceService.getTypingUsers(conversationId);
        return ResponseEntity.ok(typingUsers);
    }
}
