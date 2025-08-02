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
                        "/api/media/health",
                        "/api/media/actuator/**",
                        "/api/media/swagger-ui/**",
                        "/api/media/v3/api-docs/**"
                );
    }
}
