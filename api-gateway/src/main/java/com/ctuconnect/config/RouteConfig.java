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
                .route("user-service-profile-route", r -> r
                        .path("/api/users/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://user-service"))
                .route("user-service-friends-route", r -> r
                        .path("/api/users/{userId}/friends/**",
                              "/api/users/{userId}/mutual-friends/**",
                              "/api/users/{userId}/friend-suggestions")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://user-service"))
                .route("user-service-friend-request-route", r -> r
                        .path("/api/users/{userId}/invite/**",
                              "/api/users/{userId}/accept-invite/**",
                              "/api/users/{userId}/reject-invite/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://user-service"))
                .route("user-service-filter-route", r -> r
                        .path("/api/users/{userId}/filter-relationships")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://user-service"))
                .route("user-service-register-route", r -> r
                        .path("/api/users/register")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://user-service"))

                // Fallback route for any other user service endpoints
                .route("user-service-general-route", r -> r
                        .path("/api/users/**")
                        .filters(f -> f.filter(jwtAuthenticationFilter.apply(new JwtAuthenticationFilter.Config())))
                        .uri("lb://user-service"))
                .build();
    }
}
