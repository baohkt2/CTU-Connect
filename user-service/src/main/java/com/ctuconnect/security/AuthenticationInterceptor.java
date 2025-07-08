package com.ctuconnect.security;

import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/**
 * Interceptor to extract authenticated user information from request headers
 * These headers are set by the API Gateway after JWT token validation
 */
@Component
public class AuthenticationInterceptor implements HandlerInterceptor {

    private static final String USER_ID_HEADER = "X-User-Id";
    private static final String USER_EMAIL_HEADER = "X-User-Email";
    private static final String USER_ROLE_HEADER = "X-User-Role";

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        // Extract user information from headers set by gateway
        String userId = request.getHeader(USER_ID_HEADER);
        String email = request.getHeader(USER_EMAIL_HEADER);
        String role = request.getHeader(USER_ROLE_HEADER);

        // If user information is present, create authenticated user context
        if (userId != null && email != null && role != null) {
            AuthenticatedUser authenticatedUser = new AuthenticatedUser(userId, email, role);
            SecurityContextHolder.setAuthenticatedUser(authenticatedUser);
        }

        return true;
    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) {
        // Clear security context after request completion
        SecurityContextHolder.clear();
    }
}
