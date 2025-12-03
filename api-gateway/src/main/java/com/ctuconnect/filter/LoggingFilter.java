package com.ctuconnect.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

@Component
public class LoggingFilter implements GlobalFilter, Ordered {

    private static final Logger LOGGER = LoggerFactory.getLogger(LoggingFilter.class);

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();

        LOGGER.info("Request: {} {}", request.getMethod(), request.getURI());

        return chain.filter(exchange)
                .then(Mono.fromRunnable(() -> {
                    LOGGER.info("Response: {} {}", exchange.getResponse().getStatusCode(), request.getURI());
                }));
    }

    @Override
    public int getOrder() {
        // Set a high precedence to ensure this filter runs first
        return Ordered.HIGHEST_PRECEDENCE;
    }
}
