package com.ctuconnect.controller;

import com.ctuconnect.dto.*;
import com.ctuconnect.service.AuthService;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @PostMapping("/register")
    public ResponseEntity<AuthResponse> register(@Valid @RequestBody RegisterRequest request, HttpServletResponse response) {
        AuthResponse authResponse = authService.register(request);
        clearAllTokenCookies(response);
        // Set tokens in cookies
        setTokenCookies(response, authResponse.getAccessToken(), authResponse.getRefreshToken());

        // Remove tokens from response body for security
        authResponse.setAccessToken(null);
        authResponse.setRefreshToken(null);

        return ResponseEntity.ok(authResponse);
    }

    @PostMapping("/login")
    public ResponseEntity<AuthResponse> login(@Valid @RequestBody LoginRequest request, HttpServletResponse response) {
        AuthResponse authResponse = authService.login(request);
        clearAllTokenCookies(response); // Clear any existing cookies before setting new ones
        // Set tokens in cookies
        setTokenCookies(response, authResponse.getAccessToken(), authResponse.getRefreshToken());

        // Remove tokens from response body for security
        authResponse.setAccessToken(null);
        authResponse.setRefreshToken(null);

        return ResponseEntity.ok(authResponse);
    }

    @PostMapping("/refresh-token")
    public ResponseEntity<?> refreshToken(
            @CookieValue(name = "refreshToken", required = false) String refreshToken,
            HttpServletResponse response) {

        if (refreshToken == null || refreshToken.trim().isEmpty()) {
            return ResponseEntity.badRequest()
                .body(java.util.Map.of(
                    "success", false,
                    "message", "Refresh token is missing. Please login again."
                ));
        }

        try {
            AuthResponse authResponse = authService.refreshToken(refreshToken);

            clearAllTokenCookies(response);
            setTokenCookies(response, authResponse.getAccessToken(), authResponse.getRefreshToken());

            return ResponseEntity.ok(authResponse);
        } catch (RuntimeException e) {
            clearAllTokenCookies(response);
            return ResponseEntity.badRequest()
                .body(java.util.Map.of(
                    "success", false,
                    "message", e.getMessage()
                ));
        }
    }


    @PostMapping("/forgot-password")
    public ResponseEntity<String> forgotPassword(@RequestBody PasswordResetRequest request) {
        authService.forgotPassword(request.getEmail());
        return ResponseEntity.ok("Password reset email sent. Please check your email.");
    }

    @PostMapping("/reset-password")
    public ResponseEntity<String> resetPassword(@Valid @RequestBody PasswordResetRequest request) {
        authService.resetPassword(request.getToken(), request.getNewPassword());
        return ResponseEntity.ok("Password has been reset successfully.");
    }

    @PostMapping("/verify-email")
    public ResponseEntity<String> verifyEmail(@RequestParam String token) {
        authService.verifyEmail(token);
        return ResponseEntity.ok("Email verified successfully.");
    }

    @PostMapping("/resend-verification")
    public ResponseEntity<String> resendVerificationEmail(@RequestBody ResendVerificationRequest request) {
        authService.resendVerificationEmail(request.getToken());
        return ResponseEntity.ok("Verification email sent successfully.");
    }

    @PostMapping("/logout")
    public ResponseEntity<String> logout(@CookieValue(value = "refreshToken", required = false) String refreshToken, HttpServletResponse response) {
        try {
            // Only call logout service if refresh token exists
            if (refreshToken != null && !refreshToken.trim().isEmpty()) {
                authService.logout(refreshToken);
            }
        } catch (Exception e) {
            // Log error but still clear cookies
            System.err.println("Logout error: " + e.getMessage());
        }

        // Always clear all possible token cookies regardless of service call result
        clearAllTokenCookies(response);

        return ResponseEntity.ok("Logged out successfully");
    }

    @GetMapping("/me")
    public ResponseEntity<AuthResponse> getCurrentUser(
            @CookieValue(value = "accessToken", required = false) String accessToken,
            @RequestHeader(value = "X-User-Email", required = false) String userEmail,
            @RequestHeader(value = "X-User-Id", required = false) String userId) {
        
        System.out.println("AuthController /me: DEBUG - accessToken from cookie: " + accessToken);
        System.out.println("AuthController /me: DEBUG - X-User-Email header: " + userEmail);
        System.out.println("AuthController /me: DEBUG - X-User-Id header: " + userId);
        
        // Prioritize headers from API Gateway (when request comes through gateway)
        if (userEmail != null && !userEmail.trim().isEmpty()) {
            System.out.println("AuthController /me: Using email from header: " + userEmail);
            try {
                AuthResponse authResponse = authService.getCurrentUserByEmail(userEmail);
                return ResponseEntity.ok(authResponse);
            } catch (Exception e) {
                System.err.println("AuthController /me: Error getting user by email: " + e.getMessage());
                return ResponseEntity.status(401).build();
            }
        }
        
        // Fallback to cookie-based authentication (direct calls)
        if (accessToken == null || accessToken.trim().isEmpty()) {
            System.err.println("AuthController /me: No authentication credentials found");
            return ResponseEntity.status(401).build();
        }

        try {
            AuthResponse authResponse = authService.getCurrentUser(accessToken);
            return ResponseEntity.ok(authResponse);
        } catch (Exception e) {
            System.err.println("AuthController /me: Error with token: " + e.getMessage());
            return ResponseEntity.status(401).build();
        }
    }

    private void setTokenCookies(HttpServletResponse response, String accessToken, String refreshToken) {
        // Set access token cookie (HttpOnly for security)
        Cookie accessTokenCookie = new Cookie("accessToken", accessToken);
        accessTokenCookie.setHttpOnly(true); // Secure - JavaScript cannot access
        accessTokenCookie.setSecure(false); // Set to true in production with HTTPS
        accessTokenCookie.setPath("/");
        accessTokenCookie.setMaxAge(15 * 60); // 15 minutes

        response.addCookie(accessTokenCookie);

        // Set refresh token cookie (HttpOnly for security)
        Cookie refreshTokenCookie = new Cookie("refreshToken", refreshToken);
        refreshTokenCookie.setHttpOnly(true); // Secure - JavaScript cannot access
        refreshTokenCookie.setSecure(false); // Set to true in production with HTTPS
        refreshTokenCookie.setPath("/");
        refreshTokenCookie.setMaxAge(7 * 24 * 60 * 60); // 7 days

        response.addCookie(refreshTokenCookie);
    }

    private void clearAllTokenCookies(HttpServletResponse response) {
        // Clear new naming convention cookies
        Cookie clearAccessToken = new Cookie("accessToken", "");
        clearAccessToken.setHttpOnly(true); // Match the settings used when creating
        clearAccessToken.setSecure(false);
        clearAccessToken.setPath("/");
        clearAccessToken.setMaxAge(0);
        response.addCookie(clearAccessToken);

        Cookie clearRefreshToken = new Cookie("refreshToken", "");
        clearRefreshToken.setHttpOnly(true); // Match the settings used when creating
        clearRefreshToken.setSecure(false);
        clearRefreshToken.setPath("/");
        clearRefreshToken.setMaxAge(0);
        response.addCookie(clearRefreshToken);

        // Clear old naming convention cookies for backward compatibility
        Cookie clearOldAccessToken = new Cookie("access_token", "");
        clearOldAccessToken.setHttpOnly(true);
        clearOldAccessToken.setSecure(false);
        clearOldAccessToken.setPath("/");
        clearOldAccessToken.setMaxAge(0);
        response.addCookie(clearOldAccessToken);

        Cookie clearOldRefreshToken = new Cookie("refresh_token", "");
        clearOldRefreshToken.setHttpOnly(true);
        clearOldRefreshToken.setSecure(false);
        clearOldRefreshToken.setPath("/");
        clearOldRefreshToken.setMaxAge(0);
        response.addCookie(clearOldRefreshToken);
    }
}
