package com.ctuconnect.config;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.simp.config.ChannelRegistration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.messaging.support.ChannelInterceptor;
import org.springframework.messaging.support.MessageHeaderAccessor;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;

@Configuration
@EnableWebSocketMessageBroker
@RequiredArgsConstructor
@Slf4j
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        // Enable simple broker for sending messages to clients
        config.enableSimpleBroker("/topic", "/queue");

        // Prefix for messages that are bound for methods annotated with @MessageMapping
        config.setApplicationDestinationPrefixes("/app");

        // Prefix for user-specific destinations
        config.setUserDestinationPrefix("/user");
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        // Register STOMP endpoint for WebSocket connection
        registry.addEndpoint("/ws/chat")
                .setAllowedOriginPatterns("http://localhost:3000", "http://localhost:3001", "http://localhost:8080")
                .withSockJS();
    }

    @Override
    public void configureClientInboundChannel(ChannelRegistration registration) {
        registration.interceptors(new ChannelInterceptor() {
            @Override
            public Message<?> preSend(Message<?> message, MessageChannel channel) {
                StompHeaderAccessor accessor = MessageHeaderAccessor.getAccessor(message, StompHeaderAccessor.class);

                if (StompCommand.CONNECT.equals(accessor.getCommand())) {
                    // Extract user info from headers (passed by API Gateway after JWT validation)
                    String userId = accessor.getFirstNativeHeader("X-User-Id");
                    String username = accessor.getFirstNativeHeader("X-Username");
                    String sessionId = accessor.getSessionId();

                    if (userId != null && !userId.isEmpty()) {
                        // Set user principal from Gateway headers
                        accessor.setUser(() -> userId);
                        log.debug("User {} (username: {}) connected with session {}", userId, username, sessionId);

                        // Store user session mapping for presence tracking
                        accessor.getSessionAttributes().put("userId", userId);
                        accessor.getSessionAttributes().put("username", username);
                    } else {
                        log.warn("Missing user information in WebSocket connection headers");
                        throw new IllegalArgumentException("User authentication required");
                    }
                } else if (StompCommand.DISCONNECT.equals(accessor.getCommand())) {
                    String userId = accessor.getUser() != null ? accessor.getUser().getName() : null;
                    if (userId != null) {
                        log.debug("User {} disconnected", userId);
                        // Handle user presence cleanup
                    }
                }

                return message;
            }
        });
    }
}
