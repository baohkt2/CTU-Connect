package com.ctuconnect.config;

import com.ctuconnect.security.AuthenticationInterceptor;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Web configuration to register security interceptors
 * CORS is handled by SecurityConfig only to avoid duplicate headers
 */
@Configuration
@RequiredArgsConstructor
public class WebConfig implements WebMvcConfigurer {

    private final AuthenticationInterceptor authenticationInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(authenticationInterceptor)
                .addPathPatterns("/api/**") // Apply to all API endpoints
                .excludePathPatterns(
                        "/api/chats/health",
                        "/api/chats/actuator/**",
                        "/api/chats/swagger-ui/**",
                        "/api/chats/v3/api-docs/**"
                );
    }

    // CORS removed - handled by SecurityConfig to avoid duplicate headers
}
