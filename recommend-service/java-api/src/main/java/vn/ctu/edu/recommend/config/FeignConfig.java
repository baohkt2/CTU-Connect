package vn.ctu.edu.recommend.config;

import feign.Logger;
import feign.RequestInterceptor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

/**
 * Feign Client Configuration
 * Configures logging and request interceptors for Feign clients
 */
@Configuration
@Slf4j
public class FeignConfig {

    /**
     * Configure Feign logging level
     */
    @Bean
    public Logger.Level feignLoggerLevel() {
        return Logger.Level.BASIC;
    }

    /**
     * Request interceptor to forward JWT token to other services
     */
    @Bean
    public RequestInterceptor requestInterceptor() {
        return requestTemplate -> {
            ServletRequestAttributes attributes = 
                (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            
            if (attributes != null) {
                String authorization = attributes.getRequest().getHeader("Authorization");
                if (authorization != null && authorization.startsWith("Bearer ")) {
                    requestTemplate.header("Authorization", authorization);
                    log.debug("Forwarding Authorization header to Feign client");
                }
            }
        };
    }
}
