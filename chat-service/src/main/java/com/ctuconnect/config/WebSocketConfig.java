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
                .setAllowedOriginPatterns("http://localhost:3000", "http://localhost:3001")
                .withSockJS();
    }

    @Override
    public void configureClientInboundChannel(ChannelRegistration registration) {
        registration.interceptors(new ChannelInterceptor() {
            @Override
            public Message<?> preSend(Message<?> message, MessageChannel channel) {
                StompHeaderAccessor accessor = MessageHeaderAccessor.getAccessor(message, StompHeaderAccessor.class);

                if (StompCommand.CONNECT.equals(accessor.getCommand())) {
                    // Extract user ID from headers
                    String userId = accessor.getFirstNativeHeader("userId");
                    String sessionId = accessor.getSessionId();

                    if (userId != null && sessionId != null) {
                        accessor.setUser(() -> userId);
                        // UserPresence sẽ được handle trong Controller hoặc Service layer
                        log.debug("User {} connected with session {}", userId, sessionId);
                    }
                } else if (StompCommand.DISCONNECT.equals(accessor.getCommand())) {
                    String userId = accessor.getUser() != null ? accessor.getUser().getName() : null;
                    if (userId != null) {
                        // UserPresence sẽ được handle trong Controller hoặc Service layer
                        log.debug("User {} disconnected", userId);
                    }
                }

                return message;
            }
        });
    }
}
