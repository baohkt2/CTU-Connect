package com.ctuconnect.filter;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.validation.constraints.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferFactory;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.http.server.reactive.ServerHttpResponseDecorator;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.time.Duration;

@Component
public class AuthResponseFilter implements GlobalFilter, Ordered {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${server.ssl.enabled:false}")
    private boolean sslEnabled;

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();
        String path = request.getURI().getPath();

        System.out.println("AuthResponseFilter: Processing request for path: " + path);

        if (!path.equals("/api/auth/login") && !path.equals("/api/auth/refresh-token")) {
            System.out.println("AuthResponseFilter: Skipping non-auth endpoint: " + path);
            return chain.filter(exchange);
        }

        ServerHttpResponse originalResponse = exchange.getResponse();
        DataBufferFactory bufferFactory = originalResponse.bufferFactory();

        ServerHttpResponseDecorator decoratedResponse = new ServerHttpResponseDecorator(originalResponse) {
            @Override
            @NotNull
            public Mono<Void> writeWith(@NotNull org.reactivestreams.Publisher<? extends DataBuffer> body) {
                System.out.println("AuthResponseFilter: Entering writeWith for path: " + path);
                if (getStatusCode() != null && getStatusCode().is2xxSuccessful() && body instanceof Flux) {
                    System.out.println("AuthResponseFilter: Response is 2xx and body is Flux");
                    Flux<? extends DataBuffer> fluxBody = (Flux<? extends DataBuffer>) body;
                    return super.writeWith(fluxBody.buffer().map(dataBuffers -> {
                        DataBuffer join = bufferFactory.join(dataBuffers);
                        byte[] content = new byte[join.readableByteCount()];
                        join.read(content);
                        DataBufferUtils.release(join);

                        String responseBody = new String(content, StandardCharsets.UTF_8);
                        System.out.println("AuthResponseFilter: Raw Response Body: " + responseBody);

                        String accessToken = null;
                        String refreshToken = null;

                        // Try to get tokens from response body
                        if (!responseBody.isEmpty()) {
                            try {
                                JsonNode jsonNode = objectMapper.readTree(responseBody);
                                accessToken = jsonNode.has("accessToken") ? jsonNode.get("accessToken").asText() : null;
                                refreshToken = jsonNode.has("refreshToken") ? jsonNode.get("refreshToken").asText() : null;
                                System.out.println("AuthResponseFilter: Tokens from body - accessToken: " + (accessToken != null ? "present" : "null") + ", refreshToken: " + (refreshToken != null ? "present" : "null"));
                            } catch (Exception e) {
                                System.err.println("AuthResponseFilter: Error parsing response body: " + e.getMessage());
                            }
                        } else {
                            System.err.println("AuthResponseFilter: Response body is empty for path: " + path);
                        }

                        // Fallback to headers if tokens not found in body
                        if (accessToken == null) {
                            accessToken = getDelegate().getHeaders().getFirst("X-Access-Token");
                            System.out.println("AuthResponseFilter: Access token from header: " + (accessToken != null ? "present" : "null"));
                        }
                        if (refreshToken == null) {
                            refreshToken = getDelegate().getHeaders().getFirst("X-Refresh-Token");
                            System.out.println("AuthResponseFilter: Refresh token from header: " + (refreshToken != null ? "present" : "null"));
                        }

                        if (accessToken != null) {
                            ResponseCookie accessTokenCookie = ResponseCookie.from("access_token", accessToken)
                                    .httpOnly(true)
                                    .secure(sslEnabled)
                                    .path("/")
                                    .maxAge(Duration.ofHours(1))
                                    .sameSite("Strict")
                                    .build();
                            getDelegate().addCookie(accessTokenCookie);
                            System.out.println("AuthResponseFilter: Added 'access_token' cookie.");
                        } else {
                            System.err.println("AuthResponseFilter: No 'accessToken' found in response body or headers for path: " + path);
                        }

                        if (refreshToken != null) {
                            ResponseCookie refreshTokenCookie = ResponseCookie.from("refresh_token", refreshToken)
                                    .httpOnly(true)
                                    .secure(sslEnabled)
                                    .path("/")
                                    .maxAge(Duration.ofDays(7))
                                    .sameSite("Strict")
                                    .build();
                            getDelegate().addCookie(refreshTokenCookie);
                            System.out.println("AuthResponseFilter: Added 'refresh_token' cookie.");
                        } else {
                            System.err.println("AuthResponseFilter: No 'refreshToken' found in response body or headers for path: " + path);
                        }

                        return bufferFactory.wrap(content);
                    }));
                } else {
                    System.out.println("AuthResponseFilter: Response is not 2xx or body is not Flux for path: " + path);
                    return super.writeWith(body);
                }
            }
        };

        return chain.filter(exchange.mutate().response(decoratedResponse).build());
    }

    @Override
    public int getOrder() {
        return Ordered.LOWEST_PRECEDENCE;
    }
}