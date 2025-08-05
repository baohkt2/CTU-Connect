package com.ctuconnect.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import com.ctuconnect.dto.response.UserPresenceResponse;
import com.ctuconnect.model.UserPresence;
import com.ctuconnect.repository.UserPresenceRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class UserPresenceService {

    private final UserPresenceRepository userPresenceRepository;
    private final RedisTemplate<String, Object> redisTemplate;
    private final WebSocketService webSocketService;

    private static final String PRESENCE_CACHE_PREFIX = "presence:";
    private static final String TYPING_CACHE_PREFIX = "typing:";

    public void setUserOnline(String userId, String sessionId) {
        log.info("Setting user {} online with session {}", userId, sessionId);

        UserPresence presence = userPresenceRepository.findByUserId(userId)
            .orElse(new UserPresence());

        presence.setUserId(userId);
        presence.setStatus(UserPresence.PresenceStatus.ONLINE);
        presence.setSessionId(sessionId);
        presence.setLastSeenAt(LocalDateTime.now());
        presence.setUpdatedAt(LocalDateTime.now());

        // TODO: Lấy thông tin user từ UserService
        presence.setUserName("User " + userId);
        presence.setUserAvatar("");

        userPresenceRepository.save(presence);

        // Cache trong Redis
        cacheUserPresence(presence);

        // Broadcast presence update
        webSocketService.broadcastPresenceUpdate(convertToResponse(presence));
    }

    public void setUserOffline(String userId) {
        log.info("Setting user {} offline", userId);

        Optional<UserPresence> existingPresence = userPresenceRepository.findByUserId(userId);
        if (existingPresence.isPresent()) {
            UserPresence presence = existingPresence.get();
            presence.setStatus(UserPresence.PresenceStatus.OFFLINE);
            presence.setLastSeenAt(LocalDateTime.now());
            presence.setUpdatedAt(LocalDateTime.now());
            presence.setCurrentActivity(null);
            presence.setSessionId(null);

            userPresenceRepository.save(presence);

            // Cache trong Redis
            cacheUserPresence(presence);

            // Clear typing status
            clearTypingStatus(userId);

            // Broadcast presence update
            webSocketService.broadcastPresenceUpdate(convertToResponse(presence));
        }
    }

    public void setTypingStatus(String userId, String conversationId, boolean isTyping) {
        String typingKey = TYPING_CACHE_PREFIX + conversationId + ":" + userId;

        if (isTyping) {
            // Set typing với TTL 10 giây
            redisTemplate.opsForValue().set(typingKey, userId, 10, TimeUnit.SECONDS);

            // Update presence activity
            Optional<UserPresence> presence = userPresenceRepository.findByUserId(userId);
            if (presence.isPresent()) {
                presence.get().setCurrentActivity("typing in " + conversationId);
                presence.get().setUpdatedAt(LocalDateTime.now());
                userPresenceRepository.save(presence.get());
            }

            log.debug("User {} started typing in conversation {}", userId, conversationId);
        } else {
            // Remove typing status
            redisTemplate.delete(typingKey);
            
            // Clear presence activity
            Optional<UserPresence> presence = userPresenceRepository.findByUserId(userId);
            if (presence.isPresent()) {
                presence.get().setCurrentActivity(null);
                presence.get().setUpdatedAt(LocalDateTime.now());
                userPresenceRepository.save(presence.get());
            }

            log.debug("User {} stopped typing in conversation {}", userId, conversationId);
        }
        
        // Broadcast typing status
        webSocketService.broadcastTypingStatus(conversationId, userId, isTyping);
    }

    public List<String> getTypingUsers(String conversationId) {
        String pattern = TYPING_CACHE_PREFIX + conversationId + ":*";
        return redisTemplate.keys(pattern).stream()
            .map(key -> (String) redisTemplate.opsForValue().get(key))
            .collect(Collectors.toList());
    }

    public UserPresenceResponse getUserPresence(String userId) {
        // Thử cache trước
        UserPresence cached = getCachedUserPresence(userId);
        if (cached != null) {
            return convertToResponse(cached);
        }
        
        // Nếu không có cache, query database
        Optional<UserPresence> presence = userPresenceRepository.findByUserId(userId);
        if (presence.isPresent()) {
            cacheUserPresence(presence.get());
            return convertToResponse(presence.get());
        }
        
        // Nếu không tìm thấy, tạo presence mặc định
        return createDefaultPresence(userId);
    }

    public List<UserPresenceResponse> getMultipleUserPresence(List<String> userIds) {
        return userIds.stream()
            .map(this::getUserPresence)
            .collect(Collectors.toList());
    }

    public List<UserPresenceResponse> getOnlineUsers() {
        List<UserPresence> onlineUsers = userPresenceRepository.findByStatus(UserPresence.PresenceStatus.ONLINE);
        return onlineUsers.stream()
            .map(this::convertToResponse)
            .collect(Collectors.toList());
    }
    
    public void setUserAway(String userId) {
        log.info("Setting user {} as away", userId);

        Optional<UserPresence> existingPresence = userPresenceRepository.findByUserId(userId);
        if (existingPresence.isPresent()) {
            UserPresence presence = existingPresence.get();
            presence.setStatus(UserPresence.PresenceStatus.AWAY);
            presence.setLastSeenAt(LocalDateTime.now());
            presence.setUpdatedAt(LocalDateTime.now());
            presence.setCurrentActivity(null);

            userPresenceRepository.save(presence);

            // Cache trong Redis
            cacheUserPresence(presence);

            // Broadcast presence update
            webSocketService.broadcastPresenceUpdate(convertToResponse(presence));
        }
    }

    private void cacheUserPresence(UserPresence presence) {
        String cacheKey = PRESENCE_CACHE_PREFIX + presence.getUserId();
        redisTemplate.opsForValue().set(cacheKey, presence, 5, TimeUnit.MINUTES);
    }
    
    private UserPresence getCachedUserPresence(String userId) {
        String cacheKey = PRESENCE_CACHE_PREFIX + userId;
        return (UserPresence) redisTemplate.opsForValue().get(cacheKey);
    }
    
    private void clearTypingStatus(String userId) {
        String pattern = TYPING_CACHE_PREFIX + "*:" + userId;
        redisTemplate.keys(pattern).forEach(redisTemplate::delete);
    }
    
    private UserPresenceResponse createDefaultPresence(String userId) {
        UserPresenceResponse response = new UserPresenceResponse();
        response.setUserId(userId);
        response.setUserName("User " + userId);
        response.setUserAvatar("");
        response.setStatus(UserPresence.PresenceStatus.OFFLINE);
        response.setLastSeenAt(LocalDateTime.now());
        return response;
    }
    
    private UserPresenceResponse convertToResponse(UserPresence presence) {
        UserPresenceResponse response = new UserPresenceResponse();
        response.setUserId(presence.getUserId());
        response.setUserName(presence.getUserName());
        response.setUserAvatar(presence.getUserAvatar());
        response.setStatus(presence.getStatus());
        response.setCurrentActivity(presence.getCurrentActivity());
        response.setLastSeenAt(presence.getLastSeenAt());
        return response;
    }
}
