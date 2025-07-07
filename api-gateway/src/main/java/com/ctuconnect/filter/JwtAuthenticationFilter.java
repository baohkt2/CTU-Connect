package com.ctuconnect.filter;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import java.util.Base64;
import io.jsonwebtoken.security.Keys;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.MalformedJwtException;
import io.jsonwebtoken.UnsupportedJwtException;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.gateway.filter.GatewayFilter;
import org.springframework.cloud.gateway.filter.factory.AbstractGatewayFilterFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.security.Key;
import java.util.Date;
import java.util.List;
import java.util.Arrays;

@Component
public class JwtAuthenticationFilter extends AbstractGatewayFilterFactory<JwtAuthenticationFilter.Config> {

    // Inject secretKey from application.properties or application.yml
    @Value("${jwt.secret:XpExu6h1RJoY1qFZyLVzJbor/aYutNR2AD86ZM/tKqc=}")
    private String secretKey;

    // Paths that don't require authentication
    private final List<String> openApiEndpoints = Arrays.asList(
            "/api/auth/register",
            "/api/auth/login",
            "/api/auth/refresh-token",
            "/api/auth/forgot-password",
            "/api/auth/reset-password",
            "/api/auth/verify-email",
            "/api/test",
            "/actuator" // Actuator endpoints are often public for health checks
    );

    public JwtAuthenticationFilter() {
        super(Config.class);
    }

    @Override
    public GatewayFilter apply(Config config) {
        return (exchange, chain) -> {
            ServerHttpRequest request = exchange.getRequest();

            // Log the incoming request path for debugging
            System.out.println("Processing request path: " + request.getURI().getPath());

            // Check if the request path is an open API endpoint
            if (isOpenApiEndpoint(request)) {
                System.out.println("Path is open API endpoint, skipping authentication.");
                return chain.filter(exchange);
            }

            // Check if the request has an Authorization header
            if (!request.getHeaders().containsKey("Authorization")) {
                System.err.println("Authorization header is missing for path: " + request.getURI().getPath());
                return onError(exchange, "Authorization header is missing", HttpStatus.UNAUTHORIZED);
            }

            // Get the token from the Authorization header
            String token = request.getHeaders().getFirst("Authorization");
            if (token == null || !token.startsWith("Bearer ")) {
                System.err.println("Invalid Authorization header format for path: " + request.getURI().getPath());
                return onError(exchange, "Invalid Authorization header format", HttpStatus.UNAUTHORIZED);
            }

            // Extract the actual token string by removing "Bearer " prefix
            token = token.substring(7);

            try {
                // Validate the token and extract claims
                Claims claims = extractAllClaims(token);

                // Check if the token has expired (redundant if extractAllClaims throws ExpiredJwtException, but good for clarity)
                if (claims.getExpiration().before(new Date())) {
                    System.err.println("Token has expired for user: " + claims.getSubject());
                    return onError(exchange, "Token has expired", HttpStatus.UNAUTHORIZED);
                }

                // Add user claims to the request headers for downstream services
                // This allows microservices to easily access user ID and role
                ServerHttpRequest mutatedRequest = exchange.getRequest().mutate()
                        .header("X-Auth-User-Id", claims.getSubject())
                        .header("X-Auth-User-Role", claims.get("role", String.class)) // Assuming 'role' claim exists
                        .build();

                System.out.println("Token validated for user: " + claims.getSubject() + ", role: " + claims.get("role", String.class));
                return chain.filter(exchange.mutate().request(mutatedRequest).build());
            } catch (ExpiredJwtException e) {
                // This catch block handles expired tokens directly from JJWT parsing
                System.err.println("JWT token is expired: " + e.getMessage() + " for token: " + token);
                return onError(exchange, "JWT token is expired", HttpStatus.UNAUTHORIZED);
            } catch (MalformedJwtException e) {
                System.err.println("Invalid JWT token format: " + e.getMessage() + " for token: " + token);
                return onError(exchange, "Invalid JWT token format", HttpStatus.UNAUTHORIZED);
            } catch (UnsupportedJwtException e) {
                System.err.println("JWT token is unsupported: " + e.getMessage() + " for token: " + token);
                return onError(exchange, "Unsupported JWT token", HttpStatus.UNAUTHORIZED);
            } catch (IllegalArgumentException e) {
                System.err.println("JWT claims string is empty or malformed: " + e.getMessage() + " for token: " + token);
                return onError(exchange, "JWT claims string is empty or malformed", HttpStatus.UNAUTHORIZED);
            } catch (Exception e) {
                // Catch any other unexpected exceptions during token processing
                System.err.println("An unexpected error occurred during token validation: " + e.getMessage() + " for token: " + token);
                return onError(exchange, "An unexpected error occurred during authentication", HttpStatus.INTERNAL_SERVER_ERROR);
            }
        };
    }

    // Checks if the request URI path is in the list of open API endpoints.
    // Using `startsWith` for robustness, as `contains` could lead to false positives
    // (e.g., "/api/auth/registerUser" would match "/api/auth/register").
    private boolean isOpenApiEndpoint(ServerHttpRequest request) {
        String path = request.getURI().getPath();
        for (String uri : openApiEndpoints) {
            if (path.startsWith(uri)) {
                return true;
            }
        }
        return false;
    }

    // Handles error responses for the gateway.
    // It sets the HTTP status code and completes the response.
    private Mono<Void> onError(ServerWebExchange exchange, String message, HttpStatus status) {
        ServerHttpResponse response = exchange.getResponse();
        response.setStatusCode(status);
        // Optionally, you can add a response body with the error message
        // response.getHeaders().add("Content-Type", "application/json");
        // String errorBody = "{\"error\": \"" + message + "\"}";
        // DataBuffer buffer = response.bufferFactory().wrap(errorBody.getBytes());
        // return response.writeWith(Mono.just(buffer));
        return response.setComplete();
    }

    // Extracts all claims from a JWT token.
    // This method handles the parsing and validation of the token's signature.
    private Claims extractAllClaims(String token) {
        // Corrected: Using setSigningKey for compatibility with older JJWT versions (pre-0.12.0)
        return Jwts.parserBuilder() // parserBuilder() is available from JJWT 0.11.0+
                .setSigningKey(getSigningKey()) // Use setSigningKey()
                .build()
                .parseClaimsJws(token) // parseClaimsJws() returns a Jws<Claims>
                .getBody(); // getBody() retrieves the Claims payload
    }

    // Generates the signing key from the Base64 encoded secret string.
    private Key getSigningKey() {
        // Corrected: Using java.util.Base64 for decoding
        // This is the standard and recommended way for Java 8+
        byte[] keyBytes = Base64.getDecoder().decode(secretKey);
        return Keys.hmacShaKeyFor(keyBytes); // Creates an HMAC-SHA key from the bytes
    }



    public static class Config {
        // No configuration properties needed for this filter's current logic
        // If you wanted to make openApiEndpoints configurable per filter instance,
        // you would add properties here.
    }
}
