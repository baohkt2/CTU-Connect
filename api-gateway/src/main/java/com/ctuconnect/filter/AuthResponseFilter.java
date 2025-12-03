//package com.ctuconnect.filter;
//
//import com.fasterxml.jackson.databind.JsonNode;
//import com.fasterxml.jackson.databind.ObjectMapper;
//import jakarta.validation.constraints.NotNull;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.cloud.gateway.filter.GatewayFilterChain;
//import org.springframework.cloud.gateway.filter.GlobalFilter;
//import org.springframework.core.Ordered;
//import org.springframework.core.io.buffer.DataBuffer;
//import org.springframework.core.io.buffer.DataBufferFactory;
//import org.springframework.core.io.buffer.DataBufferUtils;
//import org.springframework.http.HttpCookie;
//import org.springframework.http.ResponseCookie;
//import org.springframework.http.server.reactive.ServerHttpRequest;
//import org.springframework.http.server.reactive.ServerHttpResponse;
//import org.springframework.http.server.reactive.ServerHttpResponseDecorator;
//import org.springframework.stereotype.Component;
//import org.springframework.web.server.ServerWebExchange;
//import reactor.core.publisher.Flux;
//import reactor.core.publisher.Mono;
//
//import java.nio.charset.StandardCharsets;
//import java.time.Duration;
//import java.util.List;
//
//@Component
//public class AuthResponseFilter implements GlobalFilter, Ordered {
//
//    private final ObjectMapper objectMapper = new ObjectMapper();
//
//    @Value("${server.ssl.enabled:false}")
//    private boolean sslEnabled;
//
//    @Override
//    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
//        ServerHttpRequest request = exchange.getRequest();
//        String path = request.getURI().getPath();
//
//        System.out.println("AuthResponseFilter: Processing request for path: " + path);
//
//        if (!path.equals("/api/auth/login") && !path.equals("/api/auth/refresh-token") && !path.equals("/api/auth/register")) {
//            System.out.println("AuthResponseFilter: Skipping non-auth endpoint: " + path);
//            return chain.filter(exchange);
//        }
//
//        ServerHttpResponse originalResponse = exchange.getResponse();
//        DataBufferFactory bufferFactory = originalResponse.bufferFactory();
//
//        ServerHttpResponseDecorator decoratedResponse = new ServerHttpResponseDecorator(originalResponse) {
//            @Override
//            @NotNull
//            public Mono<Void> writeWith(@NotNull org.reactivestreams.Publisher<? extends DataBuffer> body) {
//                System.out.println("AuthResponseFilter: Entering writeWith for path: " + path);
//                System.out.println("AuthResponseFilter: Status code: " + getStatusCode());
//                System.out.println("AuthResponseFilter: Body type: " + body.getClass().getSimpleName());
//
//                if (getStatusCode() != null && getStatusCode().is2xxSuccessful()) {
//                    System.out.println("AuthResponseFilter: Response is 2xx successful");
//
//                    if (body instanceof Flux) {
//                        System.out.println("AuthResponseFilter: Body is Flux - processing");
//
//                        // Clear old tokens first for login/register/refresh endpoints
//                        clearOldTokenCookies();
//
//                        Flux<? extends DataBuffer> fluxBody = (Flux<? extends DataBuffer>) body;
//                        return super.writeWith(fluxBody.buffer().map(dataBuffers -> {
//                            System.out.println("AuthResponseFilter: Processing dataBuffers, count: " + dataBuffers.size());
//                            DataBuffer join = bufferFactory.join(dataBuffers);
//                            byte[] content = new byte[join.readableByteCount()];
//                            join.read(content);
//                            DataBufferUtils.release(join);
//
//                            String responseBody = new String(content, StandardCharsets.UTF_8);
//                            System.out.println("AuthResponseFilter: Raw Response Body: " + responseBody);
//
//                            // Check if auth service already set cookies via Set-Cookie headers
//                            List<String> setCookieHeaders = getDelegate().getHeaders().get("Set-Cookie");
//                            boolean authServiceSetAccessToken = false;
//                            boolean authServiceSetRefreshToken = false;
//
//                            System.out.println("AuthResponseFilter: Checking Set-Cookie headers, found: " + (setCookieHeaders != null ? setCookieHeaders.size() : 0));
//
//                            if (setCookieHeaders != null) {
//                                for (String cookieHeader : setCookieHeaders) {
//                                    System.out.println("AuthResponseFilter: Found Set-Cookie header: " + cookieHeader);
//                                    if (cookieHeader.contains("accessToken=")) {
//                                        authServiceSetAccessToken = true;
//                                        System.out.println("AuthResponseFilter: Auth service set accessToken cookie");
//                                    }
//                                    if (cookieHeader.contains("refreshToken=")) {
//                                        authServiceSetRefreshToken = true;
//                                        System.out.println("AuthResponseFilter: Auth service set refreshToken cookie");
//                                    }
//                                }
//                            }
//
//                            // Only set cookies from gateway if auth service didn't set them
//                            if (!authServiceSetAccessToken || !authServiceSetRefreshToken) {
//                                System.out.println("AuthResponseFilter: Need to handle tokens from response");
//                                handleTokensFromResponse(responseBody, path, authServiceSetAccessToken, authServiceSetRefreshToken);
//                            } else {
//                                System.out.println("AuthResponseFilter: Auth service already set both tokens via cookies, skipping gateway cookie setting");
//                            }
//
//                            return bufferFactory.wrap(content);
//                        }));
//                    } else {
//                        System.out.println("AuthResponseFilter: Body is NOT Flux, type: " + body.getClass().getName() + " - using fallback");
//
//                        // Clear old tokens for non-Flux bodies too
//                        clearOldTokenCookies();
//
//                        // For non-Flux bodies, we still need to try to extract tokens
//                        // Let's try to handle this case as well
//                        return super.writeWith(body);
//                    }
//                } else {
//                    System.out.println("AuthResponseFilter: Response is not 2xx, status: " + getStatusCode() + " for path: " + path);
//                    return super.writeWith(body);
//                }
//            }
//
//            private void clearOldTokenCookies() {
//                System.out.println("AuthResponseFilter: Clearing old token cookies");
//
//                // Clear access token cookie
//                ResponseCookie clearAccessToken = ResponseCookie.from("accessToken", "")
//                        .httpOnly(true)
//                        .secure(sslEnabled)
//                        .path("/")
//                        .maxAge(Duration.ZERO)
//                        .sameSite("Strict")
//                        .build();
//                getDelegate().addCookie(clearAccessToken);
//
//                // Also clear the old naming convention if exists
//                ResponseCookie clearOldAccessToken = ResponseCookie.from("access_token", "")
//                        .httpOnly(true)
//                        .secure(sslEnabled)
//                        .path("/")
//                        .maxAge(Duration.ZERO)
//                        .sameSite("Strict")
//                        .build();
//                getDelegate().addCookie(clearOldAccessToken);
//
//                // Clear refresh token cookie
//                ResponseCookie clearRefreshToken = ResponseCookie.from("refreshToken", "")
//                        .httpOnly(true)
//                        .secure(sslEnabled)
//                        .path("/")
//                        .maxAge(Duration.ZERO)
//                        .sameSite("Strict")
//                        .build();
//                getDelegate().addCookie(clearRefreshToken);
//
//                // Also clear the old naming convention if exists
//                ResponseCookie clearOldRefreshToken = ResponseCookie.from("refresh_token", "")
//                        .httpOnly(true)
//                        .secure(sslEnabled)
//                        .path("/")
//                        .maxAge(Duration.ZERO)
//                        .sameSite("Strict")
//                        .build();
//                getDelegate().addCookie(clearOldRefreshToken);
//
//                System.out.println("AuthResponseFilter: Old token cookies cleared");
//            }
//
//            private void handleTokensFromResponse(String responseBody, String path, boolean authServiceSetAccessToken, boolean authServiceSetRefreshToken) {
//                String accessToken = null;
//                String refreshToken = null;
//
//                // Try to get tokens from response body
//                if (!responseBody.isEmpty()) {
//                    try {
//                        JsonNode jsonNode = objectMapper.readTree(responseBody);
//                        accessToken = jsonNode.has("accessToken") ? jsonNode.get("accessToken").asText() : null;
//                        refreshToken = jsonNode.has("refreshToken") ? jsonNode.get("refreshToken").asText() : null;
//                        System.out.println("AuthResponseFilter: Tokens from body - accessToken: " + (accessToken != null ? "present" : "null") + ", refreshToken: " + (refreshToken != null ? "present" : "null"));
//                    } catch (Exception e) {
//                        System.err.println("AuthResponseFilter: Error parsing response body: " + e.getMessage());
//                    }
//                } else {
//                    System.err.println("AuthResponseFilter: Response body is empty for path: " + path);
//                }
//
//                // Fallback to headers if tokens not found in body
//                if (accessToken == null) {
//                    accessToken = getDelegate().getHeaders().getFirst("X-Access-Token");
//                    System.out.println("AuthResponseFilter: Access token from header: " + (accessToken != null ? "present" : "null"));
//                }
//                if (refreshToken == null) {
//                    refreshToken = getDelegate().getHeaders().getFirst("X-Refresh-Token");
//                    System.out.println("AuthResponseFilter: Refresh token from header: " + (refreshToken != null ? "present" : "null"));
//                }
//
//                // Set new token cookies if found and not already set by auth service
//                if (accessToken != null && !accessToken.trim().isEmpty() && !authServiceSetAccessToken) {
//                    ResponseCookie accessTokenCookie = ResponseCookie.from("accessToken", accessToken)
//                            .httpOnly(true)
//                            .secure(sslEnabled)
//                            .path("/")
//                            .maxAge(Duration.ofMinutes(15)) // Match auth service expiration
//                            .sameSite("Strict")
//                            .build();
//                    getDelegate().addCookie(accessTokenCookie);
//                    System.out.println("AuthResponseFilter: Added 'accessToken' cookie from gateway.");
//                } else if (authServiceSetAccessToken) {
//                    System.out.println("AuthResponseFilter: Skipping accessToken cookie setting - already set by auth service");
//                } else {
//                    System.err.println("AuthResponseFilter: No valid 'accessToken' found for path: " + path);
//                }
//
//                if (refreshToken != null && !refreshToken.trim().isEmpty() && !authServiceSetRefreshToken) {
//                    ResponseCookie refreshTokenCookie = ResponseCookie.from("refreshToken", refreshToken)
//                            .httpOnly(true)
//                            .secure(sslEnabled)
//                            .path("/")
//                            .maxAge(Duration.ofDays(7)) // Match auth service expiration
//                            .sameSite("Strict")
//                            .build();
//                    getDelegate().addCookie(refreshTokenCookie);
//                    System.out.println("AuthResponseFilter: Added 'refreshToken' cookie from gateway.");
//                } else if (authServiceSetRefreshToken) {
//                    System.out.println("AuthResponseFilter: Skipping refreshToken cookie setting - already set by auth service");
//                } else {
//                    System.err.println("AuthResponseFilter: No valid 'refreshToken' found for path: " + path);
//                }
//            }
//        };
//
//        return chain.filter(exchange.mutate().response(decoratedResponse).build());
//    }
//
//    @Override
//    public int getOrder() {
//        return Ordered.LOWEST_PRECEDENCE;
//    }
//}
