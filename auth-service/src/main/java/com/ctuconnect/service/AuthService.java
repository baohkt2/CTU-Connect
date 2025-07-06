package com.ctuconnect.service;

import com.ctuconnect.dto.AuthResponse;
import com.ctuconnect.dto.LoginRequest;
import com.ctuconnect.dto.RegisterRequest;

public interface AuthService {

    /**
     * Register a new user
     */
    AuthResponse register(RegisterRequest request);

    /**
     * Authenticate a user and return tokens
     */
    AuthResponse login(LoginRequest request);

    /**
     * Refresh an access token using a refresh token
     */
    AuthResponse refreshToken(String refreshToken);

    /**
     * Send a password reset email
     */
    void forgotPassword(String email);

    /**
     * Reset a password using a token
     */
    void resetPassword(String token, String newPassword);

    /**
     * Verify a user's email
     */
    void verifyEmail(String token);

    /**
     * Logout a user by invalidating their refresh token
     */
    void logout(String refreshToken);
}
