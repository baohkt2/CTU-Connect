package com.ctuconnect.config;

import com.ctuconnect.filter.JwtAuthenticationFilter;
import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RouteConfig {

    private final JwtAuthenticationFilter jwtAuthenticationFilter;

    public RouteConfig(JwtAuthenticationFilter jwtAuthenticationFilter) {
        this.jwtAuthenticationFilter = jwtAuthenticationFilter;
    }

    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
                // Auth Service Routes - Public endpoints that don't require JWT validation
                .route("auth-service-route", r -> r
                        .path("/api/auth/**")
                        .uri("lb://auth-service"))

                // User Service Routes - Protected endpoints that require JWT validation
                .route("user-service-route", r -> r
                        .path("/api/users/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://user-service"))

                .route("media-service-route", r -> r
                        .path("/api/media/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://media-service"))

                .route("post-service-route", r -> r
                        .path("/api/posts/**", "/api/comments/**", "/api/search/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://post-service"))

                .route("notification-service-route", r -> r
                        .path("/api/notifications/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://notification-service"))

                .route("chat-service-route", r -> r
                        .path("/api/chats/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://chat-service"))

                // WebSocket Chat Route - Special handling for WebSocket connections
                .route("chat-websocket-route", r -> r
                        .path("/ws/chat/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://chat-service"))
                .build();
    }
}
