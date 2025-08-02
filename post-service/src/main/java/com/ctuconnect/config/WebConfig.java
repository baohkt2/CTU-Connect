package com.ctuconnect.config;

import com.ctuconnect.security.AuthenticationInterceptor;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Web configuration to register security interceptors
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
                        "/api/posts/health",
                        "/api/posts/actuator/**",
                        "/api/posts/swagger-ui/**",
                        "/api/posts/v3/api-docs/**"
                );
    }
}
