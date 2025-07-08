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

        // Set tokens in cookies
        setTokenCookies(response, authResponse.getAccessToken(), authResponse.getRefreshToken());

        // Remove tokens from response body for security
        authResponse.setAccessToken(null);
        authResponse.setRefreshToken(null);

        return ResponseEntity.ok(authResponse);
    }

    @PostMapping("/refresh-token")
    public ResponseEntity<AuthResponse> refreshToken(@Valid @RequestBody RefreshTokenRequest request, HttpServletResponse response) {
        AuthResponse authResponse = authService.refreshToken(request.getRefreshToken());

        // Set tokens in cookies
        setTokenCookies(response, authResponse.getAccessToken(), authResponse.getRefreshToken());

        // Remove tokens from response body for security
        authResponse.setAccessToken(null);
        authResponse.setRefreshToken(null);

        return ResponseEntity.ok(authResponse);
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

    @PostMapping("/logout")
    public ResponseEntity<String> logout(@Valid @RequestBody RefreshTokenRequest request, HttpServletResponse response) {
        authService.logout(request.getRefreshToken());

        // Clear cookies
        clearTokenCookies(response);

        return ResponseEntity.ok("Logged out successfully");
    }

    private void setTokenCookies(HttpServletResponse response, String accessToken, String refreshToken) {
        // Set access token cookie
        Cookie accessTokenCookie = new Cookie("accessToken", accessToken);
        accessTokenCookie.setHttpOnly(true);
        accessTokenCookie.setSecure(true); // Set to true in production with HTTPS
        accessTokenCookie.setPath("/");
        accessTokenCookie.setMaxAge(15 * 60); // 15 minutes
        response.addCookie(accessTokenCookie);

        // Set refresh token cookie
        Cookie refreshTokenCookie = new Cookie("refreshToken", refreshToken);
        refreshTokenCookie.setHttpOnly(true);
        refreshTokenCookie.setSecure(true); // Set to true in production with HTTPS
        refreshTokenCookie.setPath("/");
        refreshTokenCookie.setMaxAge(7 * 24 * 60 * 60); // 7 days
        response.addCookie(refreshTokenCookie);
    }

    private void clearTokenCookies(HttpServletResponse response) {
        // Clear access token cookie
        Cookie accessTokenCookie = new Cookie("accessToken", "");
        accessTokenCookie.setHttpOnly(true);
        accessTokenCookie.setSecure(true);
        accessTokenCookie.setPath("/");
        accessTokenCookie.setMaxAge(0);
        response.addCookie(accessTokenCookie);

        // Clear refresh token cookie
        Cookie refreshTokenCookie = new Cookie("refreshToken", "");
        refreshTokenCookie.setHttpOnly(true);
        refreshTokenCookie.setSecure(true);
        refreshTokenCookie.setPath("/");
        refreshTokenCookie.setMaxAge(0);
        response.addCookie(refreshTokenCookie);
    }
}
