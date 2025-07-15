package com.ctuconnect.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.reactive.CorsWebFilter;
import org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource;

@Configuration
public class CorsConfig {

    @Bean
    public CorsWebFilter corsWebFilter() {
        CorsConfiguration configuration = new CorsConfiguration();

        // Allow specific origins
        configuration.addAllowedOrigin("http://localhost:3000"); // Client frontend
        configuration.addAllowedOrigin("http://localhost:3001"); // Admin frontend
        configuration.addAllowedOrigin("http://localhost:8080"); // API Gateway

        // Allow all headers
        configuration.addAllowedHeader("*");

        // Allow all methods
        configuration.addAllowedMethod("*");

        // Allow credentials
        configuration.setAllowCredentials(true);

        // Set max age for preflight requests
        configuration.setMaxAge(3600L);

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);

        return new CorsWebFilter(source);
    }
}
