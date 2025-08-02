package com.ctuconnect.config;

import feign.Logger;
import feign.Request;
import feign.Retryer;
import feign.codec.ErrorDecoder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.TimeUnit;

/**
 * Feign client configuration for better error handling and retry mechanism
 */
@Configuration
public class FeignConfig {

    @Bean
    Logger.Level feignLoggerLevel() {
        return Logger.Level.BASIC;
    }

    @Bean
    public Request.Options requestOptions() {
        return new Request.Options(
                10, TimeUnit.SECONDS, // connect timeout
                60, TimeUnit.SECONDS, // read timeout
                true // follow redirects
        );
    }

    @Bean
    public Retryer retryer() {
        return new Retryer.Default(
                1000, // initial interval
                3000, // max interval
                3     // max attempts
        );
    }

    @Bean
    public ErrorDecoder errorDecoder() {
        return new CustomFeignErrorDecoder();
    }

    /**
     * Custom error decoder to handle service communication errors gracefully
     */
    public static class CustomFeignErrorDecoder implements ErrorDecoder {

        @Override
        public Exception decode(String methodKey, feign.Response response) {
            switch (response.status()) {
                case 400:
                    return new IllegalArgumentException("Bad Request: " + methodKey);
                case 404:
                    return new RuntimeException("Service not found: " + methodKey);
                case 500:
                    return new RuntimeException("Internal Server Error in " + methodKey);
                case 503:
                    return new RuntimeException("Service Unavailable: " + methodKey);
                default:
                    return new RuntimeException("Unknown error occurred in " + methodKey + ": " + response.status());
            }
        }
    }
}
