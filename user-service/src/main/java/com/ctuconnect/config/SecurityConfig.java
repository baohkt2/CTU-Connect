package com.ctuconnect.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.EnableAspectJAutoProxy;

/**
 * Configuration to enable AOP for @RequireAuth annotation processing
 */
@Configuration
@EnableAspectJAutoProxy
public class SecurityConfig {
    // AOP configuration for @RequireAuth annotation
}
